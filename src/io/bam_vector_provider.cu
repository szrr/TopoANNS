#include "topoanns/bam_vector_provider.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <memory>
#include <limits>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

#include "topoanns/common.hpp"
#include "../search/fused_rerank_device.hpp"
#include "../search/exact_distance_device.cuh"
#include "../search/topology_microbatch_io.hpp"
#include "rerank_stop_prefix_trace.hpp"

#include <ctrl.h>
#include <page_cache.h>

#include <bamWriteSSD.cuh>
#include <settings.cuh>

namespace topoanns {
namespace {

constexpr std::size_t kWarpSize = 32;
constexpr std::size_t kThreadsPerBlock = 128;
constexpr std::size_t kWarpsPerBlock = kThreadsPerBlock / kWarpSize;

bool DisableSpecializedExactDim768() {
    static const bool disabled = []() {
        const char* value = std::getenv("TOPOANNS_DISABLE_SPECIALIZED_EXACT_DIM768");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return disabled;
}
constexpr unsigned int kFullMask = 0xffffffffU;
constexpr std::size_t kMaxPersistentTileSize = 128;
constexpr std::size_t kRerankLearnedStopFeatureCount = 13;
constexpr std::size_t kRerankLearnedStopHeadProbeCount = 8;
constexpr std::size_t kMaxRerankLearnedStopStages = 64;
constexpr std::size_t kMaxRerankLearnedStopTopK = 128;
constexpr std::uint32_t kRerankLearnedStopFeatureVariantFull13 = 0U;
constexpr std::uint32_t kRerankLearnedStopFeatureVariantCore6 = 1U;
constexpr std::array<std::uint32_t, kRerankLearnedStopFeatureCount>
    kRerankLearnedStopCore6FeatureMask = {1U, 1U, 1U, 0U, 0U, 1U, 0U,
                                          1U, 0U, 1U, 0U, 0U, 0U};

struct DeviceRerankLearnedStopConfig {
    std::uint32_t enabled = 0U;
    std::uint32_t feature_variant = kRerankLearnedStopFeatureVariantFull13;
    std::uint32_t num_stages = 0U;
    std::uint32_t top_k = 0U;
    std::uint32_t bootstrap_prefix = 0U;
    std::uint32_t initial_stage_index = 0U;
    std::uint32_t stage_prefixes[kMaxRerankLearnedStopStages] = {};
    float linear_weights[kRerankLearnedStopFeatureCount] = {};
    float bias = 0.0f;
    float threshold_logit = 0.0f;
    float budget_log2 = 0.0f;
    float inv_budget_top_n = 0.0f;
    float inv_top_k = 0.0f;
};

__constant__ DeviceRerankLearnedStopConfig kDeviceRerankLearnedStopConfig;

std::string TrimAscii(std::string text) {
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
    return text;
}

std::vector<std::string> SplitCsvTokens(const std::string& text) {
    std::vector<std::string> tokens;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        std::string token =
            text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        token = TrimAscii(std::move(token));
        if (!token.empty()) {
            tokens.push_back(std::move(token));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return tokens;
}

std::vector<std::uint32_t> ParseCsvU32(const std::string& text) {
    std::vector<std::uint32_t> values;
    for (const std::string& token : SplitCsvTokens(text)) {
        values.push_back(static_cast<std::uint32_t>(std::stoul(token)));
    }
    return values;
}

std::vector<float> ParseCsvFloat(const std::string& text) {
    std::vector<float> values;
    for (const std::string& token : SplitCsvTokens(text)) {
        values.push_back(std::stof(token));
    }
    return values;
}

DeviceRerankLearnedStopConfig LoadRerankLearnedStopConfigFromFile(
    const std::filesystem::path& path,
    std::uint32_t budget_top_n,
    std::uint32_t top_k) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error(BuildErrorMessage(
            "LoadRerankLearnedStopConfigFromFile", "Failed to open " + path.string()));
    }

    std::vector<std::uint32_t> stage_prefixes;
    std::vector<float> feature_means;
    std::vector<float> feature_inv_stds;
    std::vector<float> weights;
    std::vector<std::uint32_t> feature_mask;
    std::uint32_t file_top_k = 0U;
    std::uint32_t bootstrap_prefix = 0U;
    float bias = 0.0f;
    float threshold_logit = 0.0f;
    std::unordered_map<std::uint32_t, float> per_top_l_threshold_logits;

    std::string line;
    while (std::getline(in, line)) {
        line = TrimAscii(std::move(line));
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error(BuildErrorMessage(
                "LoadRerankLearnedStopConfigFromFile",
                "Invalid learned-stop config line: " + line));
        }
        const std::string key = TrimAscii(line.substr(0, eq));
        const std::string value = TrimAscii(line.substr(eq + 1));
        if (key == "stage_prefixes") {
            stage_prefixes = ParseCsvU32(value);
        } else if (key == "feature_means") {
            feature_means = ParseCsvFloat(value);
        } else if (key == "feature_inv_stds") {
            feature_inv_stds = ParseCsvFloat(value);
        } else if (key == "weights") {
            weights = ParseCsvFloat(value);
        } else if (key == "feature_mask") {
            feature_mask = ParseCsvU32(value);
        } else if (key == "bias") {
            bias = std::stof(value);
        } else if (key == "threshold_logit") {
            threshold_logit = std::stof(value);
        } else if (key == "top_k") {
            file_top_k = static_cast<std::uint32_t>(std::stoul(value));
        } else if (key == "bootstrap_prefix") {
            bootstrap_prefix = static_cast<std::uint32_t>(std::stoul(value));
        } else if (key.rfind("threshold_logit_top_l_", 0) == 0U) {
            const std::string suffix = key.substr(std::char_traits<char>::length("threshold_logit_top_l_"));
            if (!suffix.empty()) {
                per_top_l_threshold_logits.emplace(
                    static_cast<std::uint32_t>(std::stoul(suffix)), std::stof(value));
            }
        }
    }

    constexpr std::size_t kLegacyRerankLearnedStopFeatureCount = 12;
    const bool legacy_feature_layout =
        feature_means.size() == kLegacyRerankLearnedStopFeatureCount &&
        feature_inv_stds.size() == kLegacyRerankLearnedStopFeatureCount &&
        weights.size() == kLegacyRerankLearnedStopFeatureCount;
    const bool current_feature_layout =
        feature_means.size() == kRerankLearnedStopFeatureCount &&
        feature_inv_stds.size() == kRerankLearnedStopFeatureCount &&
        weights.size() == kRerankLearnedStopFeatureCount;
    if (!legacy_feature_layout && !current_feature_layout) {
        throw std::runtime_error(BuildErrorMessage(
            "LoadRerankLearnedStopConfigFromFile",
            "learned-stop feature vectors must all have length " +
                std::to_string(kRerankLearnedStopFeatureCount) + " or " +
                std::to_string(kLegacyRerankLearnedStopFeatureCount) + "."));
    }
    if (!feature_mask.empty() &&
        feature_mask.size() !=
            (legacy_feature_layout ? kLegacyRerankLearnedStopFeatureCount
                                   : kRerankLearnedStopFeatureCount)) {
        throw std::runtime_error(BuildErrorMessage(
            "LoadRerankLearnedStopConfigFromFile",
            "feature_mask must have length " +
                std::to_string(legacy_feature_layout ? kLegacyRerankLearnedStopFeatureCount
                                                     : kRerankLearnedStopFeatureCount) +
                "."));
    }

    DeviceRerankLearnedStopConfig config{};
    if (!legacy_feature_layout && feature_mask.size() == kRerankLearnedStopFeatureCount) {
        bool matches_core6 = true;
        for (std::size_t i = 0; i < kRerankLearnedStopFeatureCount; ++i) {
            if (feature_mask[i] != kRerankLearnedStopCore6FeatureMask[i]) {
                matches_core6 = false;
                break;
            }
        }
        if (matches_core6) {
            config.feature_variant = kRerankLearnedStopFeatureVariantCore6;
        }
    }
    config.top_k = file_top_k != 0U ? file_top_k : top_k;
    config.bootstrap_prefix = bootstrap_prefix;
    const auto per_top_l_it = per_top_l_threshold_logits.find(budget_top_n);
    config.threshold_logit =
        per_top_l_it != per_top_l_threshold_logits.end() ? per_top_l_it->second : threshold_logit;
    config.budget_log2 = std::log2(std::max(1.0f, static_cast<float>(budget_top_n)));
    config.inv_budget_top_n = budget_top_n == 0U
                                  ? 0.0f
                                  : 1.0f / static_cast<float>(budget_top_n);
    config.inv_top_k = config.top_k == 0U ? 0.0f : 1.0f / static_cast<float>(config.top_k);
    float linear_bias = bias;
    const auto add_feature = [&](std::size_t src_index, std::size_t dst_index) {
        const float masked_weight =
            (feature_mask.empty() || feature_mask[src_index] != 0U) ? weights[src_index] : 0.0f;
        config.linear_weights[dst_index] = masked_weight * feature_inv_stds[src_index];
        linear_bias -= feature_means[src_index] * config.linear_weights[dst_index];
    };
    if (legacy_feature_layout) {
        for (std::size_t i = 0; i < 5; ++i) {
            add_feature(i, i);
        }
        // Legacy models used one top-k churn ratio. The current kernel splits this into
        // topk_changed_flag and entered_topk_ratio; the ratio is the compatible signal.
        add_feature(5, 6);
        for (std::size_t i = 6; i < kLegacyRerankLearnedStopFeatureCount; ++i) {
            add_feature(i, i + 1);
        }
    } else {
        for (std::size_t i = 0; i < kRerankLearnedStopFeatureCount; ++i) {
            add_feature(i, i);
        }
    }
    config.bias = linear_bias;
    for (std::uint32_t prefix : stage_prefixes) {
        if (prefix == 0U || prefix > budget_top_n) {
            continue;
        }
        if (config.num_stages >= kMaxRerankLearnedStopStages) {
            break;
        }
        config.stage_prefixes[config.num_stages++] = prefix;
    }
    while (config.initial_stage_index < config.num_stages &&
           config.stage_prefixes[config.initial_stage_index] <= config.bootstrap_prefix) {
        ++config.initial_stage_index;
    }
    config.enabled = (config.num_stages == 0U || config.bootstrap_prefix == 0U) ? 0U : 1U;
    if (config.top_k == 0U || config.top_k > kMaxRerankLearnedStopTopK) {
        config.enabled = 0U;
    }
    return config;
}

std::uint64_t CheckedFileSize(const std::filesystem::path& path) {
    std::error_code error;
    const auto file_size = std::filesystem::file_size(path, error);
    if (error) {
        throw std::runtime_error(BuildErrorMessage("CheckedFileSize",
                                                   "Failed to stat file: " + path.string()));
    }
    return file_size;
}

bool RerankLearnedFineProfileEnabled() {
    const char* env = std::getenv("TOPOANNS_RERANK_LEARNED_FINE_PROFILE");
    if (env == nullptr) {
        return false;
    }
    const std::string value = TrimAscii(env);
    return !(value.empty() || value == "0" || value == "false" || value == "FALSE");
}

void ValidateBamLayout(const std::filesystem::path& vector_store_path,
                       std::size_t header_bytes,
                       std::size_t page_size_bytes,
                       std::size_t device_offset_bytes,
                       const std::optional<std::size_t>& payload_bytes_override = std::nullopt) {
    if (header_bytes != kDefaultPageSizeBytes) {
        throw std::runtime_error(BuildErrorMessage("ValidateBamLayout",
                                                   "BAM path expects a single 4KB header."));
    }
    if (page_size_bytes != kDefaultPageSizeBytes) {
        throw std::runtime_error(BuildErrorMessage("ValidateBamLayout",
                                                   "BAM path currently only supports 4KB pages."));
    }
    if (device_offset_bytes % page_size_bytes != 0) {
        throw std::runtime_error(BuildErrorMessage("ValidateBamLayout",
                                                   "device_offset_bytes must be 4KB aligned."));
    }
    const std::uint64_t file_size = CheckedFileSize(vector_store_path);
    if (file_size < header_bytes) {
        throw std::runtime_error(BuildErrorMessage("ValidateBamLayout",
                                                   "Vector store is smaller than its header."));
    }
    const std::uint64_t payload_bytes = file_size - header_bytes;
    const std::uint64_t effective_payload_bytes =
        payload_bytes_override.value_or(payload_bytes);
    if (effective_payload_bytes < payload_bytes) {
        throw std::runtime_error(BuildErrorMessage(
            "ValidateBamLayout", "payload_bytes_override is smaller than the file payload."));
    }
    if (effective_payload_bytes % page_size_bytes != 0) {
        throw std::runtime_error(BuildErrorMessage(
            "ValidateBamLayout", "Vector store payload must be page aligned for BAM reads."));
    }
}

std::uint64_t ResolveEffectivePayloadBytes(const std::filesystem::path& path,
                                           std::size_t header_bytes,
                                           const BamVectorProviderOptions& options) {
    const std::uint64_t file_size = CheckedFileSize(path);
    const std::uint64_t payload_bytes = file_size - header_bytes;
    return options.payload_bytes_override.value_or(payload_bytes);
}

__global__ void bam_read_combined_node_records_kernel(
    const array_d_t<char>* data_array,
    const std::uint32_t* node_ids,
    std::size_t num_nodes_to_read,
    std::size_t node_bytes,
    std::size_t nodes_per_page,
    std::uint8_t* out_records) {
    const std::size_t sample_id = blockIdx.x;
    if (sample_id >= num_nodes_to_read) {
        return;
    }

    const std::size_t lane_id = threadIdx.x;
    const std::size_t node_id = node_ids[sample_id];
    const std::size_t page_id = node_id / nodes_per_page;
    const std::size_t slot_id = node_id % nodes_per_page;
    const std::size_t page_offset = page_id * kDefaultPageSizeBytes;
    const returned_cache_page_t<char> raw_page = data_array->get_raw(page_offset);
    const auto* record = reinterpret_cast<const std::uint8_t*>(
        raw_page.addr + raw_page.offset + slot_id * node_bytes);
    std::uint8_t* output = out_records + sample_id * node_bytes;
    for (std::size_t byte = lane_id; byte < node_bytes; byte += blockDim.x) {
        output[byte] = record[byte];
    }
    __syncwarp();
    data_array->release_raw(page_offset);
}

template <typename VectorT>
__global__ void bam_fused_exact_kernel(const array_d_t<char>* data_array,
                                       const float* queries,
                                       const std::uint64_t* page_ids,
                                       const std::uint32_t* slot_ids,
                                       const std::uint32_t* node_ids,
                                       const std::uint32_t* candidate_query_ids,
                                       std::size_t page_size_bytes,
                                       std::size_t vector_bytes,
                                       std::size_t dim,
                                       std::size_t num_candidates,
                                       float* out_distances) {
    const std::size_t warp_global_id =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    if (warp_global_id >= num_candidates) {
        return;
    }

    if (node_ids[warp_global_id] == kInvalidNodeId) {
        if (lane_id == 0) {
            out_distances[warp_global_id] = __int_as_float(0x7f800000);
        }
        return;
    }

    const std::uint32_t query_id = candidate_query_ids[warp_global_id];
    const float* query = queries + static_cast<std::size_t>(query_id) * dim;
    const std::size_t page_offset =
        static_cast<std::size_t>(page_ids[warp_global_id]) * page_size_bytes;
    const returned_cache_page_t<char> raw_page = data_array->get_raw(page_offset);
    const char* page_ptr = raw_page.addr + raw_page.offset;
    const VectorT* vector = reinterpret_cast<const VectorT*>(
        page_ptr + static_cast<std::size_t>(slot_ids[warp_global_id]) * vector_bytes);

    float distance = 0.0f;
    for (std::size_t dim_idx = lane_id; dim_idx < dim; dim_idx += kWarpSize) {
        const float diff = static_cast<float>(vector[dim_idx]) - query[dim_idx];
        distance += diff * diff;
    }

    data_array->release_raw(page_offset);

    distance += __shfl_down_sync(kFullMask, distance, 16);
    distance += __shfl_down_sync(kFullMask, distance, 8);
    distance += __shfl_down_sync(kFullMask, distance, 4);
    distance += __shfl_down_sync(kFullMask, distance, 2);
    distance += __shfl_down_sync(kFullMask, distance, 1);
    if (lane_id == 0) {
        out_distances[warp_global_id] = distance;
    }
}

template <std::size_t kDim>
__global__ void bam_fused_exact_kernel_float32_dim(const array_d_t<char>* data_array,
                                                   const float* queries,
                                                   const std::uint64_t* page_ids,
                                                   const std::uint32_t* slot_ids,
                                                   const std::uint32_t* node_ids,
                                                   const std::uint32_t* candidate_query_ids,
                                                   std::size_t page_size_bytes,
                                                   std::size_t vector_bytes,
                                                   std::size_t num_candidates,
                                                   float* out_distances) {
    const std::size_t warp_global_id =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    if (warp_global_id >= num_candidates) {
        return;
    }

    if (node_ids[warp_global_id] == kInvalidNodeId) {
        if (lane_id == 0) {
            out_distances[warp_global_id] = __int_as_float(0x7f800000);
        }
        return;
    }

    const std::uint32_t query_id = candidate_query_ids[warp_global_id];
    const float* query = queries + static_cast<std::size_t>(query_id) * kDim;
    const std::size_t page_offset =
        static_cast<std::size_t>(page_ids[warp_global_id]) * page_size_bytes;
    const returned_cache_page_t<char> raw_page = data_array->get_raw(page_offset);
    const char* page_ptr = raw_page.addr + raw_page.offset;
    const float* vector = reinterpret_cast<const float*>(
        page_ptr + static_cast<std::size_t>(slot_ids[warp_global_id]) * vector_bytes);

    float distance = 0.0f;
#pragma unroll
    for (std::size_t dim_idx = lane_id; dim_idx < kDim; dim_idx += kWarpSize) {
        const float diff = vector[dim_idx] - query[dim_idx];
        distance += diff * diff;
    }

    data_array->release_raw(page_offset);

    distance += __shfl_down_sync(kFullMask, distance, 16);
    distance += __shfl_down_sync(kFullMask, distance, 8);
    distance += __shfl_down_sync(kFullMask, distance, 4);
    distance += __shfl_down_sync(kFullMask, distance, 2);
    distance += __shfl_down_sync(kFullMask, distance, 1);
    if (lane_id == 0) {
        out_distances[warp_global_id] = distance;
    }
}

inline void LaunchBamFusedExactKernelFloat32Specialized(
    const array_d_t<char>* data_array,
    const float* queries,
    const std::uint64_t* page_ids,
    const std::uint32_t* slot_ids,
    const std::uint32_t* node_ids,
    const std::uint32_t* candidate_query_ids,
    std::size_t page_size_bytes,
    std::size_t vector_bytes,
    std::size_t dim,
    std::size_t num_candidates,
    float* out_distances,
    std::size_t num_blocks) {
    switch (dim) {
        case 96:
            bam_fused_exact_kernel_float32_dim<96><<<num_blocks, kThreadsPerBlock>>>(
                data_array, queries, page_ids, slot_ids, node_ids, candidate_query_ids,
                page_size_bytes, vector_bytes, num_candidates, out_distances);
            break;
        case 128:
            bam_fused_exact_kernel_float32_dim<128><<<num_blocks, kThreadsPerBlock>>>(
                data_array, queries, page_ids, slot_ids, node_ids, candidate_query_ids,
                page_size_bytes, vector_bytes, num_candidates, out_distances);
            break;
        case 512:
            bam_fused_exact_kernel_float32_dim<512><<<num_blocks, kThreadsPerBlock>>>(
                data_array, queries, page_ids, slot_ids, node_ids, candidate_query_ids,
                page_size_bytes, vector_bytes, num_candidates, out_distances);
            break;
        case 768:
            if (DisableSpecializedExactDim768()) {
                bam_fused_exact_kernel<float><<<num_blocks, kThreadsPerBlock>>>(
                    data_array, queries, page_ids, slot_ids, node_ids, candidate_query_ids,
                    page_size_bytes, vector_bytes, dim, num_candidates, out_distances);
            } else {
                bam_fused_exact_kernel_float32_dim<768><<<num_blocks, kThreadsPerBlock>>>(
                    data_array, queries, page_ids, slot_ids, node_ids, candidate_query_ids,
                    page_size_bytes, vector_bytes, num_candidates, out_distances);
            }
            break;
        default:
            bam_fused_exact_kernel<float><<<num_blocks, kThreadsPerBlock>>>(
                data_array, queries, page_ids, slot_ids, node_ids, candidate_query_ids,
                page_size_bytes, vector_bytes, dim, num_candidates, out_distances);
            break;
    }
}

__device__ inline bool PersistentRerankLess(float lhs_distance,
                                            std::uint32_t lhs_node_id,
                                            float rhs_distance,
                                            std::uint32_t rhs_node_id) {
    if (lhs_distance != rhs_distance) {
        return lhs_distance < rhs_distance;
    }
    return lhs_node_id < rhs_node_id;
}

__device__ inline float PersistentSquaredL2LowerBound(float approx_sq_distance,
                                                      float error_bound_l2) {
    const float approx_l2 = sqrtf(fmaxf(approx_sq_distance, 0.0f));
    const float lower_l2 = fmaxf(0.0f, approx_l2 - error_bound_l2);
    return lower_l2 * lower_l2;
}

__device__ __forceinline__ float EvaluateRerankLearnedStopLogitFast(
    const DeviceRerankLearnedStopConfig& config,
    std::uint32_t processed_prefix,
    float current_kth,
    float delta_kth_last,
    float delta_kth_prev,
    float topk_changed_flag,
    std::uint32_t entered_topk_count,
    float topk_spread,
    std::uint32_t next_window_live_count,
    std::uint32_t next_window_count,
    float best_live_lb,
    std::uint32_t window_exact_count,
    std::uint32_t window_candidate_count,
    std::uint32_t cumulative_exact_count) {
    const float prefix_frac =
        static_cast<float>(processed_prefix) * config.inv_budget_top_n;
    const float entered_topk_ratio =
        static_cast<float>(entered_topk_count) * config.inv_top_k;
    const float next_window_live_ratio =
        next_window_count == 0U
            ? 0.0f
            : static_cast<float>(next_window_live_count) /
                  static_cast<float>(next_window_count);
    const float next_window_best_lb_gap =
        isfinite(best_live_lb) ? (best_live_lb - current_kth) : 0.0f;
    const float inv_window_count =
        window_candidate_count == 0U
            ? 0.0f
            : 1.0f / static_cast<float>(window_candidate_count);
    const float current_window_exact_ratio =
        static_cast<float>(window_exact_count) * inv_window_count;
    const float current_window_filtered_ratio =
        static_cast<float>(window_candidate_count - window_exact_count) * inv_window_count;
    const float cumulative_exact_frac =
        processed_prefix == 0U
            ? 0.0f
            : static_cast<float>(cumulative_exact_count) /
                  static_cast<float>(processed_prefix);

    float logit = config.bias;
    logit = fmaf(config.budget_log2, config.linear_weights[0], logit);
    logit = fmaf(prefix_frac, config.linear_weights[1], logit);
    logit = fmaf(current_kth, config.linear_weights[2], logit);
    logit = fmaf(delta_kth_last, config.linear_weights[3], logit);
    logit = fmaf(delta_kth_prev, config.linear_weights[4], logit);
    logit = fmaf(topk_changed_flag, config.linear_weights[5], logit);
    logit = fmaf(entered_topk_ratio, config.linear_weights[6], logit);
    logit = fmaf(topk_spread, config.linear_weights[7], logit);
    logit = fmaf(next_window_live_ratio, config.linear_weights[8], logit);
    logit = fmaf(next_window_best_lb_gap, config.linear_weights[9], logit);
    logit = fmaf(current_window_exact_ratio, config.linear_weights[10], logit);
    logit = fmaf(current_window_filtered_ratio, config.linear_weights[11], logit);
    return fmaf(cumulative_exact_frac, config.linear_weights[12], logit);
}

__device__ __forceinline__ float EvaluateRerankLearnedStopLogitFastCore6(
    const DeviceRerankLearnedStopConfig& config,
    std::uint32_t processed_prefix,
    float current_kth,
    float topk_changed_flag,
    float topk_spread,
    float best_live_lb) {
    const float prefix_frac =
        static_cast<float>(processed_prefix) * config.inv_budget_top_n;
    const float next_window_best_lb_gap =
        isfinite(best_live_lb) ? (best_live_lb - current_kth) : 0.0f;

    float logit = config.bias;
    logit = fmaf(config.budget_log2, config.linear_weights[0], logit);
    logit = fmaf(prefix_frac, config.linear_weights[1], logit);
    logit = fmaf(current_kth, config.linear_weights[2], logit);
    logit = fmaf(topk_changed_flag, config.linear_weights[5], logit);
    logit = fmaf(topk_spread, config.linear_weights[7], logit);
    return fmaf(next_window_best_lb_gap, config.linear_weights[9], logit);
}

__device__ inline bool PersistentInsertIntoSortedTopN(float distance,
                                                      std::uint32_t node_id,
                                                      float* distances,
                                                      std::uint32_t* node_ids,
                                                      std::size_t top_n) {
    if (top_n == 0) {
        return false;
    }
    if (!PersistentRerankLess(distance, node_id, distances[top_n - 1], node_ids[top_n - 1])) {
        return false;
    }

    std::size_t insert = top_n - 1;
    while (insert > 0 &&
           PersistentRerankLess(distance, node_id, distances[insert - 1], node_ids[insert - 1])) {
        distances[insert] = distances[insert - 1];
        node_ids[insert] = node_ids[insert - 1];
        --insert;
    }
    distances[insert] = distance;
    node_ids[insert] = node_id;
    return true;
}

__global__ void precompute_candidate_lower_bounds_kernel(
    const detail::DeviceTopologyCandidate* topology_candidates,
    std::size_t candidate_capacity,
    std::size_t budget_top_n,
    std::size_t num_queries,
    const float* error_bounds,
    const float* query_norm_squares,
    float* out_lower_bounds) {
    const std::size_t flat_idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t total_active = budget_top_n * num_queries;
    if (flat_idx >= total_active) {
        return;
    }
    const std::size_t query_id = flat_idx / budget_top_n;
    const std::size_t candidate_rank = flat_idx % budget_top_n;
    const std::size_t candidate_index = query_id * candidate_capacity + candidate_rank;
    const detail::DeviceTopologyCandidate candidate = topology_candidates[candidate_index];
    float lower_bound = std::numeric_limits<float>::infinity();
    if (candidate.valid()) {
        const float approx_sq_distance =
            candidate.distance - query_norm_squares[query_id];
        lower_bound = PersistentSquaredL2LowerBound(
            approx_sq_distance, error_bounds[candidate.raw_node_id()]);
    }
    out_lower_bounds[candidate_index] = lower_bound;
}

__device__ inline void PersistentSortTile(float* distances,
                                          std::uint32_t* node_ids,
                                          std::size_t count) {
    for (std::size_t i = 1; i < count; ++i) {
        const float distance = distances[i];
        const std::uint32_t node_id = node_ids[i];
        std::size_t insert = i;
        while (insert > 0 &&
               PersistentRerankLess(distance, node_id, distances[insert - 1], node_ids[insert - 1])) {
            distances[insert] = distances[insert - 1];
            node_ids[insert] = node_ids[insert - 1];
            --insert;
        }
        distances[insert] = distance;
        node_ids[insert] = node_id;
    }
}

__device__ __forceinline__ std::uint32_t HashExactReuseNodeForRerank(
    std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__device__ __forceinline__ bool LookupExactReuse(
    const std::uint32_t* node_ids,
    const float* distances,
    std::uint32_t capacity,
    std::uint32_t node_id,
    float* out_distance) {
    const std::uint32_t mask = capacity - 1U;
    const std::uint32_t start = HashExactReuseNodeForRerank(node_id) & mask;
    for (std::uint32_t probe = 0; probe < capacity; ++probe) {
        const std::uint32_t slot = (start + probe) & mask;
        const std::uint32_t cached_node_id = node_ids[slot];
        if (cached_node_id == node_id) {
            *out_distance = distances[slot];
            return true;
        }
        if (cached_node_id == kInvalidNodeId) {
            return false;
        }
    }
    return false;
}

template <std::size_t kDim, bool kUseBoundFilter>
__global__ void persistent_bam_rerank_kernel_float32_dim(
    const array_d_t<char>* data_array,
    const float* queries,
    const detail::DeviceTopologyCandidate* topology_candidates,
    std::size_t candidate_capacity,
    std::size_t budget_top_n,
    std::size_t result_top_n,
    std::size_t rank_tile_size,
    std::size_t vectors_per_page,
    std::size_t page_size_bytes,
    std::size_t vector_bytes,
    std::size_t top_k,
    bool use_early_stop,
    std::size_t early_stop_min_prefix,
    std::size_t early_stop_patience_tiles,
    bool use_learned_schedule,
    bool evaluate_learned_stop,
    bool enable_learned_fine_profile,
    const std::uint32_t* replay_stop_prefixes,
    const float* error_bounds,
    const float* query_norm_squares,
    const float* precomputed_lower_bounds,
    const std::uint32_t* exact_reuse_node_ids,
    const float* exact_reuse_distances,
    std::size_t exact_reuse_capacity,
    float* out_distances,
    std::uint32_t* out_node_ids,
    std::uint32_t* out_exact_counts,
    std::uint32_t* out_rerank_ssd_io_counts,
    std::uint32_t* out_reused_exact_counts,
    std::uint32_t* out_filtered_counts,
    std::uint64_t* out_learned_cycles,
    std::uint64_t* out_checkpoint_bookkeeping_cycles,
    std::uint64_t* out_topk_churn_cycles,
    std::uint64_t* out_next_window_scan_cycles,
    std::uint64_t* out_logit_eval_cycles,
    std::uint64_t* out_exact_reuse_lookup_cycles,
    std::uint64_t* out_query_block_cycles,
    std::uint32_t* out_checkpoint_counts,
    std::uint32_t* out_stop_prefixes,
    std::uint32_t* out_stop_flags) {
    const std::size_t query_id = blockIdx.x;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    const std::size_t query_candidate_base = query_id * candidate_capacity;
    const std::size_t query_result_base = query_id * result_top_n;
    const DeviceRerankLearnedStopConfig& learned_stop = kDeviceRerankLearnedStopConfig;

    __shared__ float shared_tile_distances[kMaxPersistentTileSize];
    __shared__ std::uint32_t shared_tile_node_ids[kMaxPersistentTileSize];
    __shared__ float shared_current_kth;
    __shared__ std::uint32_t shared_exact_count;
    __shared__ std::uint32_t shared_rerank_ssd_io_count;
    __shared__ std::uint32_t shared_reused_exact_count;
    __shared__ unsigned long long shared_exact_reuse_lookup_cycles;
    __shared__ std::uint32_t shared_filtered_count;
    __shared__ std::uint32_t shared_processed_prefix;
    __shared__ std::uint32_t shared_tile_valid_count;
    __shared__ std::uint32_t shared_stable_tiles;
    __shared__ std::uint32_t shared_stage_index;
    __shared__ std::uint32_t shared_next_prefix;
    __shared__ std::uint32_t shared_checkpoint_count;
    __shared__ std::uint32_t shared_stop_prefix;
    __shared__ std::uint32_t shared_has_previous_checkpoint;
    __shared__ float shared_previous_checkpoint_kth;
    __shared__ float shared_previous_delta_kth;
    __shared__ std::uint32_t shared_entered_topk_count;
    __shared__ std::uint32_t shared_checkpoint_due;
    __shared__ std::uint32_t shared_feature_next_window_begin;
    __shared__ std::uint32_t shared_feature_next_window_end;
    __shared__ float shared_topk_changed_flag;
    __shared__ float shared_topk_spread;
    __shared__ float shared_delta_kth_last;
    __shared__ std::uint32_t shared_scan_counts[kThreadsPerBlock];
    __shared__ float shared_scan_bests[kThreadsPerBlock];
    __shared__ std::uint64_t shared_feature_begin_cycle;
    __shared__ std::uint64_t shared_scan_start_cycle;
    __shared__ std::uint64_t shared_learned_cycles;
    __shared__ std::uint64_t shared_checkpoint_bookkeeping_cycles;
    __shared__ std::uint64_t shared_topk_churn_cycles;
    __shared__ std::uint64_t shared_next_window_scan_cycles;
    __shared__ std::uint64_t shared_logit_eval_cycles;
    __shared__ std::uint64_t shared_query_block_begin_cycle;
    __shared__ int shared_stop;

    const float* query = queries + query_id * kDim;

    if (threadIdx.x == 0) {
        shared_current_kth = std::numeric_limits<float>::infinity();
        shared_exact_count = 0;
        shared_rerank_ssd_io_count = 0;
        shared_reused_exact_count = 0;
        shared_exact_reuse_lookup_cycles = 0;
        shared_filtered_count = 0;
        shared_processed_prefix = 0;
        shared_query_block_begin_cycle = clock64();
        shared_tile_valid_count = 0U;
        shared_stable_tiles = 0;
        shared_stage_index = learned_stop.initial_stage_index;
        shared_next_prefix =
            use_learned_schedule
                ? min(budget_top_n, static_cast<std::size_t>(learned_stop.bootstrap_prefix))
                : min(budget_top_n, rank_tile_size);
        shared_checkpoint_count = 0;
        shared_stop_prefix = static_cast<std::uint32_t>(budget_top_n);
        shared_has_previous_checkpoint = 0U;
        shared_previous_checkpoint_kth = std::numeric_limits<float>::infinity();
        shared_previous_delta_kth = 0.0f;
        shared_entered_topk_count = 0U;
        shared_checkpoint_due = 0U;
        shared_feature_next_window_begin = 0U;
        shared_feature_next_window_end = 0U;
        shared_topk_changed_flag = 0.0f;
        shared_topk_spread = 0.0f;
        shared_delta_kth_last = 0.0f;
        shared_feature_begin_cycle = 0U;
        shared_scan_start_cycle = 0U;
        shared_learned_cycles = 0U;
        shared_checkpoint_bookkeeping_cycles = 0U;
        shared_topk_churn_cycles = 0U;
        shared_next_window_scan_cycles = 0U;
        shared_logit_eval_cycles = 0U;
        shared_stop = 0;
    }
    for (std::size_t rank = threadIdx.x; rank < result_top_n; rank += blockDim.x) {
        out_distances[query_result_base + rank] = std::numeric_limits<float>::infinity();
        out_node_ids[query_result_base + rank] = kInvalidNodeId;
    }
    __syncthreads();

    for (std::size_t rank_begin = 0; rank_begin < budget_top_n;) {
        if (shared_stop != 0) {
            break;
        }

        const std::size_t rank_end =
            use_learned_schedule ? shared_next_prefix : min(budget_top_n, rank_begin + rank_tile_size);
        const std::size_t rank_count = rank_end - rank_begin;
        if (threadIdx.x < rank_count) {
            shared_tile_distances[threadIdx.x] = std::numeric_limits<float>::infinity();
            shared_tile_node_ids[threadIdx.x] = kInvalidNodeId;
        }
        __syncthreads();

        const float query_threshold = shared_current_kth;
        const bool threshold_ready = kUseBoundFilter && isfinite(query_threshold);
        for (std::size_t local_rank = warp_id; local_rank < rank_count;
             local_rank += kWarpsPerBlock) {
            const detail::DeviceTopologyCandidate candidate =
                topology_candidates[query_candidate_base + rank_begin + local_rank];
            if (!candidate.valid()) {
                if (lane_id == 0) {
                    shared_tile_distances[local_rank] = std::numeric_limits<float>::infinity();
                    shared_tile_node_ids[local_rank] = kInvalidNodeId;
                }
                continue;
            }

            bool keep = true;
            if constexpr (kUseBoundFilter) {
                if (threshold_ready) {
                    const float lower_bound =
                        precomputed_lower_bounds[query_candidate_base + rank_begin + local_rank];
                    if (lower_bound > query_threshold) {
                        keep = false;
                    }
                }
            }
            if (!keep) {
                if (lane_id == 0) {
                    atomicAdd(&shared_filtered_count, 1U);
                    shared_tile_distances[local_rank] = std::numeric_limits<float>::infinity();
                    shared_tile_node_ids[local_rank] = kInvalidNodeId;
                }
                continue;
            }

            const std::uint32_t raw_node_id = candidate.raw_node_id();

            if (exact_reuse_node_ids != nullptr && exact_reuse_distances != nullptr &&
                exact_reuse_capacity != 0) {
                bool reused = false;
                float reused_distance = 0.0f;
                if (lane_id == 0) {
                    const std::uint64_t lookup_begin = clock64();
                    const std::size_t cache_base = query_id * exact_reuse_capacity;
                    reused = LookupExactReuse(
                        exact_reuse_node_ids + cache_base,
                        exact_reuse_distances + cache_base,
                        static_cast<std::uint32_t>(exact_reuse_capacity), raw_node_id,
                        &reused_distance);
                    atomicAdd(&shared_exact_reuse_lookup_cycles,
                              clock64() - lookup_begin);
                }
                const int reused_flag =
                    __shfl_sync(kFullMask, reused ? 1 : 0, 0);
                reused_distance = __shfl_sync(kFullMask, reused_distance, 0);
                if (reused_flag != 0) {
                    if (lane_id == 0) {
                        atomicAdd(&shared_exact_count, 1U);
                        atomicAdd(&shared_reused_exact_count, 1U);
                        shared_tile_distances[local_rank] = reused_distance;
                        shared_tile_node_ids[local_rank] = raw_node_id;
                    }
                    continue;
                }
            }

            const std::size_t page_id =
                static_cast<std::size_t>(raw_node_id) / vectors_per_page;
            const std::size_t page_offset = page_id * page_size_bytes;
            const returned_cache_page_t<char> raw_page = data_array->get_raw(page_offset);
            const char* page_ptr = raw_page.addr + raw_page.offset;
            const float* vector = reinterpret_cast<const float*>(
                page_ptr + static_cast<std::size_t>(raw_node_id % vectors_per_page) *
                               vector_bytes);

            const float distance =
                detail::WarpFloat32SquaredL2<kDim>(vector, query, lane_id);
            data_array->release_raw(page_offset);

            if (lane_id == 0) {
                atomicAdd(&shared_exact_count, 1U);
                atomicAdd(&shared_rerank_ssd_io_count, 1U);
                shared_tile_distances[local_rank] = distance;
                shared_tile_node_ids[local_rank] = raw_node_id;
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            std::size_t tile_valid_count = 0;
            for (std::size_t local_rank = 0; local_rank < rank_count; ++local_rank) {
                if (shared_tile_node_ids[local_rank] == kInvalidNodeId) {
                    continue;
                }
                shared_tile_distances[tile_valid_count] = shared_tile_distances[local_rank];
                shared_tile_node_ids[tile_valid_count] = shared_tile_node_ids[local_rank];
                ++tile_valid_count;
            }
            PersistentSortTile(shared_tile_distances, shared_tile_node_ids, tile_valid_count);
            shared_tile_valid_count = static_cast<std::uint32_t>(tile_valid_count);

            const float previous_window_kth = out_distances[query_result_base + top_k - 1];
            std::uint32_t entered_topk_count = 0U;
            for (std::size_t tile_index = 0; tile_index < tile_valid_count; ++tile_index) {
                if (PersistentInsertIntoSortedTopN(shared_tile_distances[tile_index],
                                                   shared_tile_node_ids[tile_index],
                                                   out_distances + query_result_base,
                                                   out_node_ids + query_result_base,
                                                   result_top_n)) {
                    ++entered_topk_count;
                }
            }

            shared_processed_prefix = static_cast<std::uint32_t>(rank_end);
            const float current_kth = out_distances[query_result_base + top_k - 1];
            shared_current_kth = current_kth;
            shared_checkpoint_due = 0U;
            if (evaluate_learned_stop && learned_stop.enabled != 0U && isfinite(current_kth) &&
                shared_stage_index < learned_stop.num_stages &&
                shared_processed_prefix >= learned_stop.stage_prefixes[shared_stage_index]) {
                const std::uint64_t feature_begin = clock64();
                const std::uint64_t bookkeeping_begin =
                    enable_learned_fine_profile ? clock64() : 0U;
                const float delta_kth_last =
                    shared_has_previous_checkpoint != 0U
                        ? (shared_previous_checkpoint_kth - current_kth)
                        : 0.0f;
                shared_delta_kth_last = delta_kth_last;
                shared_topk_spread = current_kth - out_distances[query_result_base];
                shared_entered_topk_count = entered_topk_count;
                shared_topk_changed_flag = entered_topk_count == 0U ? 0.0f : 1.0f;
                std::size_t next_window_begin = rank_end;
                std::size_t next_window_end = rank_end;
                if (shared_stage_index + 1U < learned_stop.num_stages) {
                    next_window_end = min(
                        budget_top_n,
                        static_cast<std::size_t>(
                            learned_stop.stage_prefixes[shared_stage_index + 1U]));
                } else {
                    next_window_end = min(budget_top_n, rank_end + rank_tile_size);
                }
                if (next_window_end < next_window_begin) {
                    next_window_end = next_window_begin;
                }
                next_window_end =
                    min(next_window_end, next_window_begin + kRerankLearnedStopHeadProbeCount);
                const bool use_next_window_features =
                    learned_stop.linear_weights[8] != 0.0f ||
                    learned_stop.linear_weights[9] != 0.0f;
                const std::size_t next_window_count =
                    next_window_end - next_window_begin;

                shared_feature_begin_cycle = feature_begin;
                shared_feature_next_window_begin =
                    static_cast<std::uint32_t>(next_window_begin);
                shared_feature_next_window_end =
                    static_cast<std::uint32_t>(next_window_end);
                if (enable_learned_fine_profile) {
                    const std::uint64_t bookkeeping_end = clock64();
                    shared_checkpoint_bookkeeping_cycles +=
                        (bookkeeping_end - bookkeeping_begin);
                }

                if (!use_next_window_features || next_window_count <= 2U) {
                    const std::uint64_t scan_begin =
                        enable_learned_fine_profile ? clock64() : 0U;
                    std::size_t next_window_live_count = 0;
                    float best_live_lb = std::numeric_limits<float>::infinity();
                    if (use_next_window_features) {
                        for (std::size_t i = 0; i < next_window_count; ++i) {
                            const float lower_bound =
                                precomputed_lower_bounds[query_candidate_base +
                                                         next_window_begin + i];
                            if (lower_bound <= current_kth) {
                                ++next_window_live_count;
                                best_live_lb = fminf(best_live_lb, lower_bound);
                            }
                        }
                    }
                if (enable_learned_fine_profile) {
                    const std::uint64_t scan_end = clock64();
                    shared_next_window_scan_cycles += (scan_end - scan_begin);
                }

                const std::uint64_t logit_begin =
                    enable_learned_fine_profile ? clock64() : 0U;
                float logit = 0.0f;
                if (learned_stop.feature_variant ==
                    kRerankLearnedStopFeatureVariantCore6) {
                    logit = EvaluateRerankLearnedStopLogitFastCore6(
                        learned_stop,
                        shared_processed_prefix,
                        current_kth,
                        shared_topk_changed_flag,
                        shared_topk_spread,
                        best_live_lb);
                } else {
                    logit = EvaluateRerankLearnedStopLogitFast(
                        learned_stop,
                        shared_processed_prefix,
                        current_kth,
                        shared_delta_kth_last,
                        shared_previous_delta_kth,
                        shared_topk_changed_flag,
                        shared_entered_topk_count,
                        shared_topk_spread,
                        static_cast<std::uint32_t>(next_window_live_count),
                        static_cast<std::uint32_t>(next_window_count),
                        best_live_lb,
                        shared_tile_valid_count,
                        static_cast<std::uint32_t>(rank_count),
                        shared_exact_count);
                }
                const std::uint64_t logit_end = clock64();
                if (enable_learned_fine_profile) {
                    shared_logit_eval_cycles += (logit_end - logit_begin);
                }
                    shared_learned_cycles += (logit_end - shared_feature_begin_cycle);
                    ++shared_checkpoint_count;
                    if (logit < learned_stop.threshold_logit) {
                        shared_stop = 1;
                        shared_stop_prefix = shared_processed_prefix;
                    }
                    shared_previous_checkpoint_kth = shared_current_kth;
                    shared_previous_delta_kth = shared_delta_kth_last;
                    shared_has_previous_checkpoint = 1U;
                    ++shared_stage_index;

                    if (shared_stop == 0) {
                        if (shared_stage_index < learned_stop.num_stages) {
                            shared_next_prefix = min(
                                budget_top_n,
                                static_cast<std::size_t>(
                                    learned_stop.stage_prefixes[shared_stage_index]));
                        } else {
                            shared_next_prefix =
                                min(budget_top_n, shared_processed_prefix + rank_tile_size);
                        }
                    }
                    shared_checkpoint_due = 0U;
                    shared_scan_start_cycle = 0U;
                } else {
                    if (enable_learned_fine_profile) {
                        shared_scan_start_cycle = clock64();
                    } else {
                        shared_scan_start_cycle = 0U;
                    }
                    shared_checkpoint_due = 1U;
                }
            } else if (replay_stop_prefixes != nullptr) {
                const std::uint32_t replay_prefix = replay_stop_prefixes[query_id];
                while (shared_stage_index < learned_stop.num_stages &&
                       learned_stop.stage_prefixes[shared_stage_index] <=
                           shared_processed_prefix) {
                    ++shared_stage_index;
                }
                if (shared_processed_prefix >= replay_prefix) {
                    shared_stop = 1;
                    shared_stop_prefix = shared_processed_prefix;
                }
            } else if (use_early_stop && shared_processed_prefix >= early_stop_min_prefix &&
                       isfinite(current_kth)) {
                if (current_kth < previous_window_kth) {
                    shared_stable_tiles = 0;
                } else {
                    ++shared_stable_tiles;
                }
                if (shared_stable_tiles >= early_stop_patience_tiles) {
                    shared_stop = 1;
                    shared_stop_prefix = shared_processed_prefix;
                }
            }

            if (use_learned_schedule && shared_stop == 0 && shared_checkpoint_due == 0U) {
                if (shared_stage_index < learned_stop.num_stages) {
                    shared_next_prefix = min(
                        budget_top_n,
                        static_cast<std::size_t>(learned_stop.stage_prefixes[shared_stage_index]));
                } else {
                    shared_next_prefix =
                        min(budget_top_n, shared_processed_prefix + rank_tile_size);
                }
            }
        }
        __syncthreads();

        const bool use_next_window_features =
            evaluate_learned_stop && shared_checkpoint_due != 0U &&
            (learned_stop.linear_weights[8] != 0.0f ||
             learned_stop.linear_weights[9] != 0.0f);
        const std::size_t next_window_begin =
            static_cast<std::size_t>(shared_feature_next_window_begin);
        const std::size_t next_window_end =
            static_cast<std::size_t>(shared_feature_next_window_end);
        const std::size_t next_window_count = next_window_end - next_window_begin;
        if (evaluate_learned_stop && shared_checkpoint_due != 0U) {
            if (next_window_count <= 8U) {
                std::size_t local_live_count = 0;
                float local_best_live_lb = std::numeric_limits<float>::infinity();
                if (warp_id == 0 && lane_id < next_window_count) {
                    const float lower_bound = precomputed_lower_bounds[query_candidate_base +
                                                                       next_window_begin + lane_id];
                    if (lower_bound <= shared_current_kth) {
                        local_live_count = 1;
                        local_best_live_lb = lower_bound;
                    }
                }
                if (warp_id == 0) {
                    for (int offset = 16; offset > 0; offset >>= 1) {
                        local_live_count +=
                            static_cast<std::size_t>(__shfl_down_sync(
                                kFullMask, static_cast<unsigned int>(local_live_count), offset));
                        local_best_live_lb = fminf(
                            local_best_live_lb,
                            __shfl_down_sync(kFullMask, local_best_live_lb, offset));
                    }
                    if (lane_id == 0) {
                        shared_scan_counts[0] =
                            static_cast<std::uint32_t>(local_live_count);
                        shared_scan_bests[0] = local_best_live_lb;
                    }
                }
            } else {
                std::size_t local_live_count = 0;
                float local_best_live_lb = std::numeric_limits<float>::infinity();
                for (std::size_t candidate_rank = next_window_begin + threadIdx.x;
                     candidate_rank < next_window_end; candidate_rank += blockDim.x) {
                    const float lower_bound =
                        precomputed_lower_bounds[query_candidate_base + candidate_rank];
                    if (lower_bound <= shared_current_kth) {
                        ++local_live_count;
                        local_best_live_lb = fminf(local_best_live_lb, lower_bound);
                    }
                }
                shared_scan_counts[threadIdx.x] =
                    static_cast<std::uint32_t>(local_live_count);
                shared_scan_bests[threadIdx.x] = local_best_live_lb;
                __syncthreads();
                for (std::uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
                    if (threadIdx.x < stride) {
                        shared_scan_counts[threadIdx.x] +=
                            shared_scan_counts[threadIdx.x + stride];
                        shared_scan_bests[threadIdx.x] =
                            fminf(shared_scan_bests[threadIdx.x],
                                  shared_scan_bests[threadIdx.x + stride]);
                    }
                    __syncthreads();
                }
            }

            if (threadIdx.x == 0) {
                const std::size_t next_window_live_count = shared_scan_counts[0];
                const float best_live_lb = shared_scan_bests[0];
                if (enable_learned_fine_profile) {
                    const std::uint64_t next_window_end_cycles = clock64();
                    shared_next_window_scan_cycles +=
                        (next_window_end_cycles - shared_scan_start_cycle);
                }

                const std::uint64_t logit_begin =
                    enable_learned_fine_profile ? clock64() : 0U;
                float logit = 0.0f;
                if (learned_stop.feature_variant ==
                    kRerankLearnedStopFeatureVariantCore6) {
                    logit = EvaluateRerankLearnedStopLogitFastCore6(
                        learned_stop,
                        shared_processed_prefix,
                        shared_current_kth,
                        shared_topk_changed_flag,
                        shared_topk_spread,
                        best_live_lb);
                } else {
                    logit = EvaluateRerankLearnedStopLogitFast(
                        learned_stop,
                        shared_processed_prefix,
                        shared_current_kth,
                        shared_delta_kth_last,
                        shared_previous_delta_kth,
                        shared_topk_changed_flag,
                        shared_entered_topk_count,
                        shared_topk_spread,
                        static_cast<std::uint32_t>(next_window_live_count),
                        static_cast<std::uint32_t>(next_window_count),
                        best_live_lb,
                        shared_tile_valid_count,
                        static_cast<std::uint32_t>(rank_count),
                        shared_exact_count);
                }
                const std::uint64_t logit_end = clock64();
                if (enable_learned_fine_profile) {
                    shared_logit_eval_cycles += (logit_end - logit_begin);
                }
                shared_learned_cycles += (logit_end - shared_feature_begin_cycle);
                ++shared_checkpoint_count;
                if (logit < learned_stop.threshold_logit) {
                    shared_stop = 1;
                    shared_stop_prefix = shared_processed_prefix;
                }
                shared_previous_checkpoint_kth = shared_current_kth;
                shared_previous_delta_kth = shared_delta_kth_last;
                shared_has_previous_checkpoint = 1U;
                ++shared_stage_index;

                if (shared_stop == 0) {
                    if (shared_stage_index < learned_stop.num_stages) {
                        shared_next_prefix = min(
                            budget_top_n,
                            static_cast<std::size_t>(
                                learned_stop.stage_prefixes[shared_stage_index]));
                    } else {
                        shared_next_prefix =
                            min(budget_top_n, shared_processed_prefix + rank_tile_size);
                    }
                }
                shared_checkpoint_due = 0U;
            }
        }
        __syncthreads();

        rank_begin = rank_end;
    }

    if (threadIdx.x == 0) {
        out_exact_counts[query_id] = shared_exact_count;
        out_rerank_ssd_io_counts[query_id] = shared_rerank_ssd_io_count;
        out_reused_exact_counts[query_id] = shared_reused_exact_count;
        out_exact_reuse_lookup_cycles[query_id] = shared_exact_reuse_lookup_cycles;
        out_filtered_counts[query_id] = shared_filtered_count;
        if (out_query_block_cycles != nullptr) {
            out_query_block_cycles[query_id] = clock64() - shared_query_block_begin_cycle;
        }
        if (evaluate_learned_stop) {
            if (out_learned_cycles != nullptr) {
                out_learned_cycles[query_id] = shared_learned_cycles;
            }
            if (out_checkpoint_bookkeeping_cycles != nullptr) {
                out_checkpoint_bookkeeping_cycles[query_id] =
                    shared_checkpoint_bookkeeping_cycles;
            }
            if (out_topk_churn_cycles != nullptr) {
                out_topk_churn_cycles[query_id] = shared_topk_churn_cycles;
            }
            if (out_next_window_scan_cycles != nullptr) {
                out_next_window_scan_cycles[query_id] = shared_next_window_scan_cycles;
            }
            if (out_logit_eval_cycles != nullptr) {
                out_logit_eval_cycles[query_id] = shared_logit_eval_cycles;
            }
            if (out_checkpoint_counts != nullptr) {
                out_checkpoint_counts[query_id] = shared_checkpoint_count;
            }
            if (out_stop_prefixes != nullptr) {
                out_stop_prefixes[query_id] = shared_stop_prefix;
            }
            if (out_stop_flags != nullptr) {
                out_stop_flags[query_id] = shared_stop != 0 ? 1U : 0U;
            }
        }
    }
}

template <bool kUseBoundFilter>
void LaunchPersistentBamRerankKernelFloat32Specialized(
    std::size_t dim,
    std::size_t num_queries,
    cudaStream_t stream,
    const array_d_t<char>* data_array,
    const float* queries,
    const detail::DeviceTopologyCandidate* topology_candidates,
    std::size_t candidate_capacity,
    std::size_t budget_top_n,
    std::size_t result_top_n,
    std::size_t rank_tile_size,
    std::size_t vectors_per_page,
    std::size_t page_size_bytes,
    std::size_t vector_bytes,
    std::size_t top_k,
    bool use_early_stop,
    std::size_t early_stop_min_prefix,
    std::size_t early_stop_patience_tiles,
    bool use_learned_schedule,
    bool evaluate_learned_stop,
    bool enable_learned_fine_profile,
    const std::uint32_t* replay_stop_prefixes,
    const float* error_bounds,
    const float* query_norm_squares,
    const float* precomputed_lower_bounds,
    const std::uint32_t* exact_reuse_node_ids,
    const float* exact_reuse_distances,
    std::size_t exact_reuse_capacity,
    float* out_distances,
    std::uint32_t* out_node_ids,
    std::uint32_t* out_exact_counts,
    std::uint32_t* out_rerank_ssd_io_counts,
    std::uint32_t* out_reused_exact_counts,
    std::uint32_t* out_filtered_counts,
    std::uint64_t* out_learned_cycles,
    std::uint64_t* out_checkpoint_bookkeeping_cycles,
    std::uint64_t* out_topk_churn_cycles,
    std::uint64_t* out_next_window_scan_cycles,
    std::uint64_t* out_logit_eval_cycles,
    std::uint64_t* out_exact_reuse_lookup_cycles,
    std::uint64_t* out_query_block_cycles,
    std::uint32_t* out_checkpoint_counts,
    std::uint32_t* out_stop_prefixes,
    std::uint32_t* out_stop_flags) {
    switch (dim) {
        case 96:
            persistent_bam_rerank_kernel_float32_dim<96, kUseBoundFilter>
                <<<num_queries, kThreadsPerBlock, 0, stream>>>(
                    data_array, queries, topology_candidates, candidate_capacity, budget_top_n,
                    result_top_n, rank_tile_size, vectors_per_page, page_size_bytes,
                    vector_bytes, top_k, use_early_stop, early_stop_min_prefix,
                    early_stop_patience_tiles, use_learned_schedule, evaluate_learned_stop,
                    enable_learned_fine_profile, replay_stop_prefixes, error_bounds, query_norm_squares, precomputed_lower_bounds, exact_reuse_node_ids, exact_reuse_distances,
                    exact_reuse_capacity, out_distances,
                    out_node_ids, out_exact_counts, out_rerank_ssd_io_counts,
                    out_reused_exact_counts, out_filtered_counts, out_learned_cycles,
                    out_checkpoint_bookkeeping_cycles, out_topk_churn_cycles,
                    out_next_window_scan_cycles, out_logit_eval_cycles, out_exact_reuse_lookup_cycles,
                    out_query_block_cycles, out_checkpoint_counts,
                    out_stop_prefixes, out_stop_flags);
            break;
        case 128:
            persistent_bam_rerank_kernel_float32_dim<128, kUseBoundFilter>
                <<<num_queries, kThreadsPerBlock, 0, stream>>>(
                    data_array, queries, topology_candidates, candidate_capacity, budget_top_n,
                    result_top_n, rank_tile_size, vectors_per_page, page_size_bytes,
                    vector_bytes, top_k, use_early_stop, early_stop_min_prefix,
                    early_stop_patience_tiles, use_learned_schedule, evaluate_learned_stop,
                    enable_learned_fine_profile, replay_stop_prefixes, error_bounds, query_norm_squares, precomputed_lower_bounds, exact_reuse_node_ids, exact_reuse_distances,
                    exact_reuse_capacity, out_distances,
                    out_node_ids, out_exact_counts, out_rerank_ssd_io_counts,
                    out_reused_exact_counts, out_filtered_counts, out_learned_cycles,
                    out_checkpoint_bookkeeping_cycles, out_topk_churn_cycles,
                    out_next_window_scan_cycles, out_logit_eval_cycles, out_exact_reuse_lookup_cycles,
                    out_query_block_cycles, out_checkpoint_counts,
                    out_stop_prefixes, out_stop_flags);
            break;
        case 512:
            persistent_bam_rerank_kernel_float32_dim<512, kUseBoundFilter>
                <<<num_queries, kThreadsPerBlock, 0, stream>>>(
                    data_array, queries, topology_candidates, candidate_capacity, budget_top_n,
                    result_top_n, rank_tile_size, vectors_per_page, page_size_bytes,
                    vector_bytes, top_k, use_early_stop, early_stop_min_prefix,
                    early_stop_patience_tiles, use_learned_schedule, evaluate_learned_stop,
                    enable_learned_fine_profile, replay_stop_prefixes, error_bounds, query_norm_squares, precomputed_lower_bounds, exact_reuse_node_ids, exact_reuse_distances,
                    exact_reuse_capacity, out_distances,
                    out_node_ids, out_exact_counts, out_rerank_ssd_io_counts,
                    out_reused_exact_counts, out_filtered_counts, out_learned_cycles,
                    out_checkpoint_bookkeeping_cycles, out_topk_churn_cycles,
                    out_next_window_scan_cycles, out_logit_eval_cycles, out_exact_reuse_lookup_cycles,
                    out_query_block_cycles, out_checkpoint_counts,
                    out_stop_prefixes, out_stop_flags);
            break;
        case 768:
            persistent_bam_rerank_kernel_float32_dim<768, kUseBoundFilter>
                <<<num_queries, kThreadsPerBlock, 0, stream>>>(
                    data_array, queries, topology_candidates, candidate_capacity, budget_top_n,
                    result_top_n, rank_tile_size, vectors_per_page, page_size_bytes,
                    vector_bytes, top_k, use_early_stop, early_stop_min_prefix,
                    early_stop_patience_tiles, use_learned_schedule, evaluate_learned_stop,
                    enable_learned_fine_profile, replay_stop_prefixes, error_bounds, query_norm_squares, precomputed_lower_bounds, exact_reuse_node_ids, exact_reuse_distances,
                    exact_reuse_capacity, out_distances,
                    out_node_ids, out_exact_counts, out_rerank_ssd_io_counts,
                    out_reused_exact_counts, out_filtered_counts, out_learned_cycles,
                    out_checkpoint_bookkeeping_cycles, out_topk_churn_cycles,
                    out_next_window_scan_cycles, out_logit_eval_cycles, out_exact_reuse_lookup_cycles,
                    out_query_block_cycles, out_checkpoint_counts,
                    out_stop_prefixes, out_stop_flags);
            break;
        default:
            throw std::runtime_error(BuildErrorMessage(
                "RunPersistentBamRerankFloat32",
                "Persistent rerank currently supports specialized float32 dimensions 96, 128, 512, and 768."));
    }
}

template <typename VectorT>
double RunBamFusedExactDistanceFloatQueriesImpl(
    const VectorPageProvider& provider,
    const CudaBuffer<float>& device_queries,
    const CudaBuffer<std::uint64_t>& page_ids,
    const CudaBuffer<std::uint32_t>& slot_ids,
    const CudaBuffer<std::uint32_t>& node_ids,
    const CudaBuffer<std::uint32_t>& candidate_query_ids,
    std::size_t num_candidates,
    const VectorPageLayout& layout,
    std::size_t dim,
    CudaBuffer<float>* out_distances) {
    const auto* data_array = static_cast<const array_d_t<char>*>(provider.DeviceReadHandle());
    if (data_array == nullptr) {
        throw std::runtime_error(BuildErrorMessage(
            "RunBamFusedExactDistanceFloatQueries",
            "Provider does not expose a BaM device handle."));
    }

    *out_distances = CudaBuffer<float>::Allocate(num_candidates);
    if (num_candidates == 0) {
        return 0.0;
    }

    const std::size_t num_blocks = (num_candidates + kWarpsPerBlock - 1) / kWarpsPerBlock;
    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin), "cudaEventRecord");
    if constexpr (std::is_same_v<VectorT, float>) {
        LaunchBamFusedExactKernelFloat32Specialized(
            data_array, device_queries.get(), page_ids.get(), slot_ids.get(), node_ids.get(),
            candidate_query_ids.get(), layout.page_size_bytes(), layout.vector_bytes(), dim,
            num_candidates, out_distances->get(), num_blocks);
    } else {
        bam_fused_exact_kernel<VectorT><<<num_blocks, kThreadsPerBlock>>>(
            data_array, device_queries.get(), page_ids.get(), slot_ids.get(), node_ids.get(),
            candidate_query_ids.get(), layout.page_size_bytes(), layout.vector_bytes(), dim,
            num_candidates, out_distances->get());
    }
    ThrowIfCudaError(cudaGetLastError(), "bam_fused_exact_kernel");
    ThrowIfCudaError(cudaEventRecord(kernel_end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);
    return static_cast<double>(kernel_ms);
}

Settings MakeBamSettings(const BamVectorProviderOptions& options,
                         std::size_t page_size_bytes) {
    Settings settings;
    settings.cudaDevice = options.cuda_device;
    settings.nvmNamespace = options.nvm_namespace;
    settings.queueDepth = options.queue_depth;
    settings.numQueues = options.num_queues;
    settings.pageSize = page_size_bytes;
    settings.maxPageCacheSize = options.page_cache_size_bytes;
    settings.ofileoffset = options.device_offset_bytes;
    return settings;
}

std::runtime_error BuildBamInitError(const char* context,
                                     const std::filesystem::path& controller_path,
                                     std::uint32_t cuda_device,
                                     const std::exception& cause) {
    std::string message = "Failed to initialize BAM controller " +
                          controller_path.string() + " for cuda_device=" +
                          std::to_string(cuda_device) + ": " + cause.what();
    const std::string cause_text = cause.what();
    if (cause_text.find("cudaHostRegister") != std::string::npos &&
        cause_text.find("operation not permitted") != std::string::npos) {
        message +=
            " Hint: this host rejects cudaHostRegisterIoMemory for the current process."
            " In the legacy GAS scripts, BAM queries are launched with elevated privileges"
            " (sudo). If you are not running as root, rerun the BAM path with elevated privileges."
            " To target a specific GPU, pass --bam-cuda-device <index>"
            " or set CUDA_VISIBLE_DEVICES=<device index> and use logical cuda:0.";
        if (geteuid() != 0) {
            message += " Current effective uid is non-root.";
        }
    }
    return std::runtime_error(BuildErrorMessage(context, message));
}

double RecordPrecomputeLowerBounds(
    const detail::DeviceTopologyCandidate* topology_candidates,
    std::size_t candidate_capacity,
    std::size_t budget_top_n,
    std::size_t num_queries,
    const float* error_bounds,
    const float* query_norm_squares,
    float* out_lower_bounds,
    cudaStream_t stream = 0) {
    const std::size_t total_active = budget_top_n * num_queries;
    if (total_active == 0) {
        return 0.0;
    }
    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin, stream), "cudaEventRecord");
    const std::size_t threads = 256;
    const std::size_t blocks = (total_active + threads - 1) / threads;
    precompute_candidate_lower_bounds_kernel<<<blocks, threads, 0, stream>>>(
        topology_candidates, candidate_capacity, budget_top_n, num_queries, error_bounds,
        query_norm_squares, out_lower_bounds);
    ThrowIfCudaError(cudaGetLastError(), "precompute_candidate_lower_bounds_kernel");
    ThrowIfCudaError(cudaEventRecord(kernel_end, stream), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);
    return static_cast<double>(kernel_ms);
}

double CyclesToMilliseconds(std::uint64_t cycles) {
    if (cycles == 0) {
        return 0.0;
    }
    int device = 0;
    ThrowIfCudaError(cudaGetDevice(&device), "cudaGetDevice");
    int clock_rate_khz = 0;
    ThrowIfCudaError(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device),
                     "cudaDeviceGetAttribute(cudaDevAttrClockRate)");
    if (clock_rate_khz <= 0) {
        return 0.0;
    }
    return static_cast<double>(cycles) / static_cast<double>(clock_rate_khz);
}

__global__ void bam_4k_trace_benchmark_kernel(
    const array_d_t<char>* data_array,
    const std::uint64_t* page_ids,
    std::size_t start_index,
    std::size_t num_requests,
    std::size_t page_size_bytes,
    unsigned long long* checksum,
    std::uint32_t* block_sm_ids) {
    if (threadIdx.x == 0) {
        block_sm_ids[blockIdx.x] = get_smid();
    }
    const std::size_t lane = threadIdx.x % kWarpSize;
    const std::size_t global_warp =
        (blockIdx.x * blockDim.x + threadIdx.x) / kWarpSize;
    const std::size_t total_warps =
        (gridDim.x * blockDim.x) / kWarpSize;
    unsigned long long local_checksum = 0;

    for (std::size_t request = global_warp; request < num_requests;
         request += total_warps) {
        const std::uint64_t page_id = page_ids[start_index + request];
        const std::size_t byte_offset = page_id * page_size_bytes;
        const returned_cache_page_t<char> raw_page = data_array->get_raw(byte_offset);
        if (lane == 0) {
            local_checksum += static_cast<unsigned char>(raw_page.addr[raw_page.offset]);
        }
        __syncwarp();
        data_array->release_raw(byte_offset);
    }
    if (lane == 0) {
        atomicAdd(checksum, local_checksum);
    }
}

__global__ void bam_4k_thread_trace_benchmark_kernel(
    const array_d_t<char>* data_array,
    const std::uint64_t* page_ids,
    std::size_t start_index,
    std::size_t num_requests,
    std::size_t page_size_bytes,
    unsigned long long* checksum,
    std::uint32_t* block_sm_ids) {
    if (threadIdx.x == 0) {
        block_sm_ids[blockIdx.x] = get_smid();
    }
    const std::size_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t total_threads = gridDim.x * blockDim.x;
    unsigned long long local_checksum = 0;

    for (std::size_t request = global_thread; request < num_requests;
         request += total_threads) {
        const std::uint64_t page_id = page_ids[start_index + request];
        const std::size_t byte_offset = page_id * page_size_bytes;
        const returned_cache_page_t<char> raw_page = data_array->get_raw(byte_offset);
        local_checksum += static_cast<unsigned char>(raw_page.addr[raw_page.offset]);
        data_array->release_raw(byte_offset);
    }
    if (local_checksum != 0) {
        atomicAdd(checksum, local_checksum);
    }
}

}  // namespace

namespace detail {
namespace {

constexpr std::size_t kWarpSize = 32;
constexpr std::size_t kMaxThreads = 256;

template <std::size_t kDim>
__device__ float ExactDistance(const float* vector,
                               const float* query,
                               std::uint32_t lane_id) {
    return WarpFloat32SquaredL2<kDim>(vector, query, lane_id);
}

__global__ void BamTopologyMicrobatchReadKernel(
    const array_d_t<char>* data_array,
    const DeviceTopologyIoRequest* requests,
    const std::uint32_t* request_count,
    std::size_t request_capacity,
    TopologyMicrobatchIoConfig config,
    std::uint32_t* response_neighbors,
    float* response_exact_distances) {
    __shared__ const char* record_ptrs[kMaxThreads];
    __shared__ std::size_t read_offsets[kMaxThreads];
    __shared__ DeviceTopologyIoRequest staged_requests[kMaxThreads];

    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    const std::size_t warps_per_block = blockDim.x / kWarpSize;
    const std::size_t total_requests =
        min(static_cast<std::size_t>(*request_count), request_capacity);
    const bool combined = config.combined_node_bytes != 0U;
    const std::size_t node_bytes =
        combined
            ? static_cast<std::size_t>(config.combined_node_bytes)
            : static_cast<std::size_t>(config.degree) * sizeof(std::uint32_t);
    const std::size_t nodes_per_page =
        combined
            ? static_cast<std::size_t>(config.combined_nodes_per_page)
            : static_cast<std::size_t>(config.topology_nodes_per_page);
    const std::size_t vector_bytes =
        static_cast<std::size_t>(config.query_dim) * sizeof(float);

    for (std::size_t tile_begin = blockIdx.x * blockDim.x;
         tile_begin < total_requests;
         tile_begin += gridDim.x * blockDim.x) {
        const std::size_t request_index = tile_begin + threadIdx.x;
        const bool active = request_index < total_requests;
        if (active) {
            const DeviceTopologyIoRequest request = requests[request_index];
            const std::size_t page_id =
                static_cast<std::size_t>(request.node_id) / nodes_per_page;
            const std::size_t slot_id =
                static_cast<std::size_t>(request.node_id) - page_id * nodes_per_page;
            const std::size_t read_offset =
                page_id * kDefaultPageSizeBytes + slot_id * node_bytes;
            const returned_cache_page_t<char> raw_page =
                data_array->get_raw(read_offset);
            record_ptrs[threadIdx.x] = raw_page.addr + raw_page.offset;
            read_offsets[threadIdx.x] = read_offset;
            staged_requests[threadIdx.x] = request;
        }
        __syncthreads();

        const std::size_t tile_count =
            min(static_cast<std::size_t>(blockDim.x), total_requests - tile_begin);
        for (std::size_t local_request = warp_id;
             local_request < tile_count;
             local_request += warps_per_block) {
            const DeviceTopologyIoRequest request = staged_requests[local_request];
            const char* record_ptr = record_ptrs[local_request];
            const std::uint32_t* neighbors = nullptr;
            std::uint32_t record_degree = config.degree;
            float exact_distance = std::numeric_limits<float>::infinity();
            if (combined) {
                const float* vector = reinterpret_cast<const float*>(record_ptr);
                const auto* record_degree_ptr =
                    reinterpret_cast<const std::uint32_t*>(record_ptr + vector_bytes);
                record_degree = *record_degree_ptr;
                neighbors = record_degree_ptr + 1;
                const float* query =
                    config.queries +
                    static_cast<std::size_t>(request.global_query_id) *
                        config.query_dim;
                switch (config.query_dim) {
                    case 96:
                        exact_distance = ExactDistance<96>(
                            vector, query, static_cast<std::uint32_t>(lane_id));
                        break;
                    case 128:
                        exact_distance = ExactDistance<128>(
                            vector, query, static_cast<std::uint32_t>(lane_id));
                        break;
                    case 512:
                        exact_distance = ExactDistance<512>(
                            vector, query, static_cast<std::uint32_t>(lane_id));
                        break;
                    case 768:
                        exact_distance = ExactDistance<768>(
                            vector, query, static_cast<std::uint32_t>(lane_id));
                        break;
                    default:
                        break;
                }
            } else {
                neighbors = reinterpret_cast<const std::uint32_t*>(record_ptr);
            }

            std::uint32_t* destination =
                response_neighbors +
                static_cast<std::size_t>(request.response_index) * config.degree;
            for (std::size_t neighbor_index = lane_id;
                 neighbor_index < config.degree;
                 neighbor_index += kWarpSize) {
                const std::uint32_t neighbor =
                    neighbor_index < record_degree &&
                            neighbors[neighbor_index] < config.num_nodes
                        ? neighbors[neighbor_index]
                        : kInvalidNodeId;
                destination[neighbor_index] = neighbor;
                if (config.validation_topology != nullptr &&
                    config.validation_mismatch_neighbors != nullptr) {
                    const std::uint32_t expected =
                        config.validation_topology[
                            static_cast<std::size_t>(request.node_id) *
                                config.degree +
                            neighbor_index];
                    if (neighbor != expected) {
                        atomicAdd(config.validation_mismatch_neighbors, 1ULL);
                    }
                }
            }
            if (lane_id == 0) {
                response_exact_distances[request.response_index] = exact_distance;
            }
        }
        __syncthreads();

        if (active) {
            data_array->release_raw(read_offsets[threadIdx.x]);
        }
        __syncthreads();
    }
}

}  // namespace

void LaunchBamTopologyMicrobatchReads(
    const void* device_read_handle,
    const DeviceTopologyIoRequest* requests,
    const std::uint32_t* request_count,
    std::size_t request_capacity,
    const TopologyMicrobatchIoConfig& config,
    std::uint32_t* response_neighbors,
    float* response_exact_distances,
    std::size_t num_blocks,
    std::size_t threads_per_block,
    cudaStream_t stream) {
    if (device_read_handle == nullptr || requests == nullptr ||
        request_count == nullptr || response_neighbors == nullptr ||
        response_exact_distances == nullptr || request_capacity == 0 ||
        num_blocks == 0 || threads_per_block == 0 ||
        threads_per_block > kMaxThreads || threads_per_block % kWarpSize != 0 ||
        config.degree == 0 || config.num_nodes == 0) {
        throw std::invalid_argument(
            "invalid BaM topology microbatch launch configuration");
    }
    const bool combined = config.combined_node_bytes != 0U;
    if ((combined &&
         (config.combined_nodes_per_page == 0 || config.queries == nullptr ||
          (config.query_dim != 96 && config.query_dim != 128 &&
           config.query_dim != 512 && config.query_dim != 768))) ||
        (!combined && config.topology_nodes_per_page == 0)) {
        throw std::invalid_argument(
            "invalid BaM topology microbatch record layout");
    }
    const auto* data_array =
        static_cast<const array_d_t<char>*>(device_read_handle);
    BamTopologyMicrobatchReadKernel
        <<<num_blocks, threads_per_block, 0, stream>>>(
            data_array, requests, request_count, request_capacity, config,
            response_neighbors, response_exact_distances);
    ThrowIfCudaError(cudaGetLastError(), "BamTopologyMicrobatchReadKernel");
}

__device__ void CopyTopologyNeighborsFromBam(const void* topology_ssd_handle,
                                             std::size_t page_offset,
                                             std::size_t node_offset_in_page,
                                             std::uint32_t degree,
                                             std::uint32_t* shared_dst,
                                             std::uint32_t lane_id,
                                             std::uint32_t active_mask) {
    const auto* data_array = static_cast<const array_d_t<char>*>(topology_ssd_handle);
    const std::size_t read_offset = page_offset + node_offset_in_page;
    const returned_cache_page_t<char> raw_page = data_array->get_raw(read_offset);
    const char* node_ptr = raw_page.addr + raw_page.offset;
    const auto* src = reinterpret_cast<const std::uint32_t*>(node_ptr);
    for (std::size_t i = lane_id; i < degree; i += kWarpSize) {
        shared_dst[i] = src[i];
    }
    static_cast<void>(active_mask);
    data_array->release_raw(read_offset);
}

__device__ float ReadCombinedNodeFromBam(const void* combined_ssd_handle,
                                         std::size_t page_offset,
                                         std::size_t node_offset_in_page,
                                         std::uint32_t vector_dim,
                                         std::uint32_t vector_bytes,
                                         std::uint64_t num_nodes,
                                         std::uint32_t degree,
                                         const float* query,
                                         std::uint32_t* shared_dst,
                                         std::uint32_t lane_id,
                                         std::uint32_t active_mask) {
    const auto* data_array = static_cast<const array_d_t<char>*>(combined_ssd_handle);
    const std::size_t read_offset = page_offset + node_offset_in_page;
    const returned_cache_page_t<char> raw_page = data_array->get_raw(read_offset);
    const char* node_ptr = raw_page.addr + raw_page.offset;
    const auto* vector = reinterpret_cast<const float*>(node_ptr);
    const auto* record_degree_ptr =
        reinterpret_cast<const std::uint32_t*>(node_ptr + vector_bytes);
    const std::uint32_t record_degree = *record_degree_ptr;
    const auto* neighbors = record_degree_ptr + 1;
    for (std::size_t i = lane_id; i < degree; i += kWarpSize) {
        shared_dst[i] = i < record_degree && neighbors[i] < num_nodes
                            ? neighbors[i]
                            : kInvalidNodeId;
    }

    float distance = 0.0f;
    switch (vector_dim) {
        case 96:
            distance = WarpFloat32SquaredL2<96>(vector, query, lane_id);
            break;
        case 128:
            distance = WarpFloat32SquaredL2<128>(vector, query, lane_id);
            break;
        case 512:
            distance = WarpFloat32SquaredL2<512>(vector, query, lane_id);
            break;
        case 768:
            distance = WarpFloat32SquaredL2<768>(vector, query, lane_id);
            break;
        default:
            distance = std::numeric_limits<float>::infinity();
            break;
    }
    static_cast<void>(active_mask);
    data_array->release_raw(read_offset);
    return distance;
}

}  // namespace detail

class BamVectorPageProvider::Impl {
public:
    Impl(const std::filesystem::path& vector_store_path,
         std::size_t header_bytes,
         std::size_t page_size_bytes,
         const BamVectorProviderOptions& options)
        : vector_store_path_(vector_store_path),
          header_bytes_(header_bytes),
          page_size_bytes_(page_size_bytes),
          options_(options),
          controller_path_string_(options.controller_path.string()) {
        if (options_.create_primary_range) {
            ValidateBamLayout(vector_store_path_, header_bytes_, page_size_bytes_,
                              options_.device_offset_bytes, options_.payload_bytes_override);
            payload_bytes_ =
                ResolveEffectivePayloadBytes(vector_store_path_, header_bytes_, options_);
            payload_pages_ = payload_bytes_ / page_size_bytes_;
        }

        settings_ = MakeBamSettings(options_, page_size_bytes_);

        try {
            controllers_.push_back(std::make_unique<Controller>(
                controller_path_string_.c_str(), settings_.nvmNamespace, settings_.cudaDevice,
                settings_.queueDepth, settings_.numQueues));
        } catch (const std::exception& e) {
            throw BuildBamInitError("BamVectorPageProvider::Impl", vector_store_path_,
                                    options_.cuda_device, e);
        }
        std::vector<Controller*> controller_ptrs = {controllers_.front().get()};

        const std::uint64_t cache_pages =
            std::max<std::uint64_t>(1, settings_.maxPageCacheSize / settings_.pageSize);
        page_cache_ = std::make_unique<page_cache_t>(
            settings_.pageSize, cache_pages, settings_.cudaDevice,
            *controller_ptrs.front(), options_.max_ranges, controller_ptrs);
        if (options_.range_id_base >= options_.max_ranges) {
            throw std::runtime_error(BuildErrorMessage(
                "BamVectorPageProvider::Impl", "range_id_base must be smaller than max_ranges."));
        }
        for (std::size_t range = 0; range < options_.range_id_base; ++range) {
            page_cache_->h_ranges[range] = nullptr;
            page_cache_->h_ranges_page_starts[range] = 0;
            page_cache_->h_ranges_dists[range] = REPLICATE;
        }
        page_cache_->pdt.n_ranges = options_.range_id_base;

        if (options_.create_primary_range) {
            const std::uint64_t page_start =
                options_.device_offset_bytes / page_size_bytes_;
            data_range_ = std::make_unique<range_t<char>>(
                0, payload_bytes_, page_start, payload_pages_, 0, page_size_bytes_,
                page_cache_.get(), settings_.cudaDevice);
            data_ranges_.push_back(data_range_.get());
            data_array_ = std::make_unique<array_t<char>>(
                payload_bytes_, options_.device_offset_bytes, data_ranges_,
                settings_.cudaDevice);
        }
    }

    const void* DeviceReadHandle() const {
        return data_array_ == nullptr ? nullptr
                                      : static_cast<const void*>(data_array_->d_array_ptr);
    }

    void FlushDevicePageCache() {
        if (page_cache_ == nullptr) {
            return;
        }
        page_cache_->flush_cache();
        ThrowIfCudaError(cudaGetLastError(), "BAM flush_cache launch");
        ThrowIfCudaError(cudaDeviceSynchronize(), "BAM flush_cache synchronize");
    }

    void ResetDevicePageCacheRanges() {
        if (page_cache_ == nullptr) {
            return;
        }
        if (options_.create_primary_range) {
            throw std::runtime_error(
                "BAM range reset is only valid for a cache-only provider");
        }
        ThrowIfCudaError(cudaDeviceSynchronize(),
                         "BAM range reset pre-synchronize");

        const std::size_t num_cache_pages = page_cache_->pdt.n_pages;
        std::unique_ptr<cache_page_t[]> cache_pages(
            new cache_page_t[num_cache_pages]);
        for (std::size_t page = 0; page < num_cache_pages; ++page) {
            cache_pages[page].page_take_lock = FREE;
            cache_pages[page].page_translation = 0;
        }
        ThrowIfCudaError(
            cudaMemcpy(page_cache_->pdt.cache_pages, cache_pages.get(),
                       num_cache_pages * sizeof(cache_page_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(BAM reset cache pages)");
        ThrowIfCudaError(
            cudaMemset(page_cache_->pdt.page_ticket, 0, sizeof(padded_struct_pc)),
            "cudaMemset(BAM reset page ticket)");

        for (std::size_t range = 0; range < options_.max_ranges; ++range) {
            page_cache_->h_ranges[range] = nullptr;
            page_cache_->h_ranges_page_starts[range] = 0;
            page_cache_->h_ranges_dists[range] = REPLICATE;
        }
        page_cache_->pdt.n_ranges = options_.range_id_base;
        ThrowIfCudaError(
            cudaMemcpy(page_cache_->pdt.ranges, page_cache_->h_ranges,
                       options_.max_ranges * sizeof(pages_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(BAM reset ranges)");
        ThrowIfCudaError(
            cudaMemcpy(page_cache_->pdt.ranges_page_starts,
                       page_cache_->h_ranges_page_starts,
                       options_.max_ranges * sizeof(std::uint64_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(BAM reset range page starts)");
        ThrowIfCudaError(
            cudaMemcpy(page_cache_->pdt.ranges_dists,
                       page_cache_->h_ranges_dists,
                       options_.max_ranges * sizeof(data_dist_t),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy(BAM reset range distributions)");
        ThrowIfCudaError(
            cudaMemcpy(page_cache_->d_pc_ptr, &page_cache_->pdt,
                       sizeof(page_cache_d_t), cudaMemcpyHostToDevice),
            "cudaMemcpy(BAM reset page cache descriptor)");
    }

    void ResetIoDepthProfile(bool enabled) {
        if (page_cache_ == nullptr) {
            return;
        }
        ThrowIfCudaError(cudaDeviceSynchronize(), "BAM I/O profile reset synchronize");
        ThrowIfCudaError(cudaMemset(page_cache_->pdt.io_submitted_cnt, 0, sizeof(std::uint64_t)), "cudaMemset(BAM submitted count)");
        ThrowIfCudaError(cudaMemset(page_cache_->pdt.io_completed_cnt, 0, sizeof(std::uint64_t)), "cudaMemset(BAM completed count)");
        ThrowIfCudaError(cudaMemset(page_cache_->pdt.io_current_outstanding, 0, sizeof(std::uint64_t)), "cudaMemset(BAM outstanding count)");
        ThrowIfCudaError(
            cudaMemset(page_cache_->pdt.io_depth_histogram, 0,
                       page_cache_->pdt.io_depth_histogram_bins * sizeof(std::uint64_t)),
            "cudaMemset(BAM depth histogram)");
        const std::uint32_t enabled_value = enabled ? 1U : 0U;
        ThrowIfCudaError(
            cudaMemcpy(page_cache_->pdt.io_depth_profile_enabled, &enabled_value,
                       sizeof(enabled_value), cudaMemcpyHostToDevice),
            "cudaMemcpy(BAM profile enabled)");
    }

    BamIoProfileSnapshot ReadIoProfile() const {
        if (page_cache_ == nullptr) {
            return {};
        }
        ThrowIfCudaError(cudaDeviceSynchronize(), "BAM I/O profile synchronize");
        BamIoProfileSnapshot snapshot;
        ThrowIfCudaError(cudaMemcpy(&snapshot.gpu_cache_hits, page_cache_->pdt.total_gpu_hit_cnt, sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy(BAM GPU hits)");
        ThrowIfCudaError(cudaMemcpy(&snapshot.host_cache_hits, page_cache_->pdt.total_host_hit_cnt, sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy(BAM host hits)");
        ThrowIfCudaError(cudaMemcpy(&snapshot.physical_reads, page_cache_->pdt.total_ssd_miss_cnt, sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy(BAM physical reads)");
        ThrowIfCudaError(cudaMemcpy(&snapshot.profiled_submissions, page_cache_->pdt.io_submitted_cnt, sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy(BAM submitted count)");
        ThrowIfCudaError(cudaMemcpy(&snapshot.profiled_completions, page_cache_->pdt.io_completed_cnt, sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy(BAM completed count)");
        ThrowIfCudaError(cudaMemcpy(&snapshot.current_outstanding, page_cache_->pdt.io_current_outstanding, sizeof(std::uint64_t), cudaMemcpyDeviceToHost), "cudaMemcpy(BAM outstanding count)");
        snapshot.submit_depth_histogram.resize(page_cache_->pdt.io_depth_histogram_bins);
        ThrowIfCudaError(
            cudaMemcpy(snapshot.submit_depth_histogram.data(), page_cache_->pdt.io_depth_histogram,
                       snapshot.submit_depth_histogram.size() * sizeof(std::uint64_t), cudaMemcpyDeviceToHost),
            "cudaMemcpy(BAM depth histogram)");
        return snapshot;
    }

    page_cache_t* SharedPageCache() const {
        return page_cache_.get();
    }

    std::uint32_t CudaDevice() const {
        return settings_.cudaDevice;
    }

    std::size_t PageSizeBytes() const {
        return page_size_bytes_;
    }

    std::vector<std::uint8_t> ReadPages(const std::filesystem::path& path,
                                        const std::vector<std::uint64_t>& page_ids,
                                        std::size_t header_bytes,
                                        std::size_t page_size_bytes) const {
        static_cast<void>(path);
        static_cast<void>(page_ids);
        static_cast<void>(header_bytes);
        static_cast<void>(page_size_bytes);
        throw std::runtime_error(BuildErrorMessage(
            "BamVectorPageProvider::ReadPages",
            "Host-side BAM page reads were removed. Use the device handle and fused BAM kernels."));
    }

    DevicePageReadResult ReadPagesToDevice(const std::filesystem::path& path,
                                           const std::vector<std::uint64_t>& page_ids,
                                           std::size_t header_bytes,
                                           std::size_t page_size_bytes) const {
        static_cast<void>(path);
        static_cast<void>(page_ids);
        static_cast<void>(header_bytes);
        static_cast<void>(page_size_bytes);
        throw std::runtime_error(BuildErrorMessage(
            "BamVectorPageProvider::ReadPagesToDevice",
            "Host-launched BAM page gathering was removed. Use the device handle and fused BAM kernels."));
    }

    DevicePageReadResult ReadPagesToDevice(const std::filesystem::path& path,
                                           const CudaBuffer<std::uint64_t>& page_ids,
                                           std::size_t num_pages,
                                           std::size_t header_bytes,
                                           std::size_t page_size_bytes) const {
        static_cast<void>(path);
        static_cast<void>(page_ids);
        static_cast<void>(num_pages);
        static_cast<void>(header_bytes);
        static_cast<void>(page_size_bytes);
        throw std::runtime_error(BuildErrorMessage(
            "BamVectorPageProvider::ReadPagesToDevice",
            "Host-launched BAM page gathering was removed. Use the device handle and fused BAM kernels."));
    }

private:
    std::filesystem::path vector_store_path_;
    std::size_t header_bytes_ = 0;
    std::size_t page_size_bytes_ = 0;
    BamVectorProviderOptions options_{};
    std::string controller_path_string_;
    Settings settings_{};
    std::uint64_t payload_bytes_ = 0;
    std::uint64_t payload_pages_ = 0;
    std::vector<std::unique_ptr<Controller>> controllers_;
    std::unique_ptr<page_cache_t> page_cache_;
    std::unique_ptr<range_t<char>> data_range_;
    std::vector<range_t<char>*> data_ranges_;
    std::unique_ptr<array_t<char>> data_array_;
};

class SharedBamFileRangeProvider final : public VectorPageProvider {
public:
    SharedBamFileRangeProvider(std::shared_ptr<BamVectorPageProvider::Impl> owner,
                               const std::filesystem::path& path,
                               std::size_t header_bytes,
                               std::size_t page_size_bytes,
                               const BamVectorProviderOptions& options)
        : owner_(std::move(owner)),
          path_(path),
          header_bytes_(header_bytes),
          page_size_bytes_(page_size_bytes),
          options_(options) {
        if (owner_ == nullptr) {
            throw std::runtime_error(BuildErrorMessage(
                "SharedBamFileRangeProvider", "owner provider is null."));
        }
        if (page_size_bytes_ != owner_->PageSizeBytes()) {
            throw std::runtime_error(BuildErrorMessage(
                "SharedBamFileRangeProvider",
                "shared BAM ranges must use the same page size as the owner provider."));
        }
        ValidateBamLayout(path_, header_bytes_, page_size_bytes_,
                          options_.device_offset_bytes, options_.payload_bytes_override);
        payload_bytes_ = ResolveEffectivePayloadBytes(path_, header_bytes_, options_);
        payload_pages_ = payload_bytes_ / page_size_bytes_;
        const std::uint64_t page_start = options_.device_offset_bytes / page_size_bytes_;
        range_ = std::make_unique<range_t<char>>(
            0, payload_bytes_, page_start, payload_pages_, 0, page_size_bytes_,
            owner_->SharedPageCache(), owner_->CudaDevice());
        ranges_.push_back(range_.get());
        array_ = std::make_unique<array_t<char>>(payload_bytes_, options_.device_offset_bytes,
                                                 ranges_, owner_->CudaDevice());
    }

    bool SupportsDeviceReads() const override { return true; }

    const void* DeviceReadHandle() const override {
        return array_ == nullptr ? nullptr : static_cast<const void*>(array_->d_array_ptr);
    }

    std::vector<std::uint8_t> ReadPages(const std::filesystem::path& path,
                                        const std::vector<std::uint64_t>& page_ids,
                                        std::size_t header_bytes,
                                        std::size_t page_size_bytes) const override {
        static_cast<void>(path);
        static_cast<void>(page_ids);
        static_cast<void>(header_bytes);
        static_cast<void>(page_size_bytes);
        throw std::runtime_error(BuildErrorMessage(
            "SharedBamFileRangeProvider::ReadPages",
            "Host-side BAM page reads were removed. Use the device handle and fused BAM kernels."));
    }

    DevicePageReadResult ReadPagesToDevice(const std::filesystem::path& path,
                                           const std::vector<std::uint64_t>& page_ids,
                                           std::size_t header_bytes,
                                           std::size_t page_size_bytes) const override {
        static_cast<void>(path);
        static_cast<void>(page_ids);
        static_cast<void>(header_bytes);
        static_cast<void>(page_size_bytes);
        throw std::runtime_error(BuildErrorMessage(
            "SharedBamFileRangeProvider::ReadPagesToDevice",
            "Host-launched BAM page gathering was removed. Use the device handle and fused BAM kernels."));
    }

    DevicePageReadResult ReadPagesToDevice(const std::filesystem::path& path,
                                           const CudaBuffer<std::uint64_t>& page_ids,
                                           std::size_t num_pages,
                                           std::size_t header_bytes,
                                           std::size_t page_size_bytes) const override {
        static_cast<void>(path);
        static_cast<void>(page_ids);
        static_cast<void>(num_pages);
        static_cast<void>(header_bytes);
        static_cast<void>(page_size_bytes);
        throw std::runtime_error(BuildErrorMessage(
            "SharedBamFileRangeProvider::ReadPagesToDevice",
            "Host-launched BAM page gathering was removed. Use the device handle and fused BAM kernels."));
    }

private:
    std::shared_ptr<BamVectorPageProvider::Impl> owner_;
    std::filesystem::path path_;
    std::size_t header_bytes_ = 0;
    std::size_t page_size_bytes_ = 0;
    BamVectorProviderOptions options_{};
    std::uint64_t payload_bytes_ = 0;
    std::uint64_t payload_pages_ = 0;
    std::unique_ptr<range_t<char>> range_;
    std::vector<range_t<char>*> ranges_;
    std::unique_ptr<array_t<char>> array_;
};

BamVectorPageProvider::BamVectorPageProvider(const std::filesystem::path& vector_store_path,
                                             std::size_t header_bytes,
                                             std::size_t page_size_bytes,
                                             const BamVectorProviderOptions& options)
    : impl_(std::make_shared<Impl>(vector_store_path, header_bytes, page_size_bytes, options)) {}

BamVectorPageProvider::~BamVectorPageProvider() = default;

BamVectorPageProvider::BamVectorPageProvider(BamVectorPageProvider&&) noexcept = default;

BamVectorPageProvider& BamVectorPageProvider::operator=(BamVectorPageProvider&&) noexcept =
    default;

std::vector<std::uint8_t> BamVectorPageProvider::ReadPages(const std::filesystem::path& path,
                                                           const std::vector<std::uint64_t>& page_ids,
                                                           std::size_t header_bytes,
                                                           std::size_t page_size_bytes) const {
    return impl_->ReadPages(path, page_ids, header_bytes, page_size_bytes);
}

DevicePageReadResult BamVectorPageProvider::ReadPagesToDevice(
    const std::filesystem::path& path,
    const std::vector<std::uint64_t>& page_ids,
    std::size_t header_bytes,
    std::size_t page_size_bytes) const {
    return impl_->ReadPagesToDevice(path, page_ids, header_bytes, page_size_bytes);
}

DevicePageReadResult BamVectorPageProvider::ReadPagesToDevice(
    const std::filesystem::path& path,
    const CudaBuffer<std::uint64_t>& page_ids,
    std::size_t num_pages,
    std::size_t header_bytes,
    std::size_t page_size_bytes) const {
    return impl_->ReadPagesToDevice(path, page_ids, num_pages, header_bytes, page_size_bytes);
}

const void* BamVectorPageProvider::DeviceReadHandle() const { return impl_->DeviceReadHandle(); }

void BamVectorPageProvider::FlushDevicePageCache() {
    impl_->FlushDevicePageCache();
}

void BamVectorPageProvider::ResetDevicePageCacheRanges() {
    impl_->ResetDevicePageCacheRanges();
}

void BamVectorPageProvider::ResetIoDepthProfile(bool enabled) {
    impl_->ResetIoDepthProfile(enabled);
}

BamIoProfileSnapshot BamVectorPageProvider::ReadIoProfile() const {
    return impl_->ReadIoProfile();
}

std::shared_ptr<VectorPageProvider> BamVectorPageProvider::CreateFileRangeProvider(
    const std::filesystem::path& path,
    std::size_t header_bytes,
    std::size_t page_size_bytes,
    const BamVectorProviderOptions& options) const {
    return std::make_shared<SharedBamFileRangeProvider>(
        impl_, path, header_bytes, page_size_bytes, options);
}


BamTraceBenchmarkWorkspace PrepareBamTraceBenchmarkWorkspace(
    std::size_t num_blocks) {
    BamTraceBenchmarkWorkspace workspace;
    workspace.device_checksum = CudaBuffer<unsigned long long>::Allocate(1);
    workspace.device_block_sm_ids = CudaBuffer<std::uint32_t>::Allocate(num_blocks);
    return workspace;
}

BamTraceBenchmarkResult RunPreparedBam4kTraceBenchmark(
    const VectorPageProvider& provider,
    const CudaBuffer<std::uint64_t>& device_page_ids,
    std::size_t start_index,
    std::size_t num_requests,
    std::size_t num_blocks,
    std::size_t warps_per_block,
    BamTraceBenchmarkWorkspace* workspace,
    cudaStream_t stream,
    std::size_t page_size_bytes,
    BamTraceRequestMapping request_mapping,
    BamTraceLaunchHostCallback launch_callback,
    void* launch_callback_context) {
    if (num_requests == 0 || num_blocks == 0 || warps_per_block == 0 ||
        warps_per_block > 32 || page_size_bytes == 0) {
        throw std::invalid_argument("invalid BaM trace benchmark dimensions");
    }
    const auto* data_array =
        static_cast<const array_d_t<char>*>(provider.DeviceReadHandle());
    if (data_array == nullptr) {
        throw std::runtime_error("BaM provider has no device read handle");
    }

    if (workspace == nullptr || workspace->device_checksum.size() != 1 ||
        workspace->device_block_sm_ids.size() != num_blocks) {
        throw std::invalid_argument("invalid BaM trace benchmark workspace");
    }
    auto& device_checksum = workspace->device_checksum;
    auto& device_block_sm_ids = workspace->device_block_sm_ids;
    ThrowIfCudaError(
        cudaMemsetAsync(device_checksum.get(), 0, sizeof(unsigned long long), stream),
        "cudaMemset(BAM trace checksum)");
    ThrowIfCudaError(
        cudaMemsetAsync(device_block_sm_ids.get(), 0xff,
                        num_blocks * sizeof(std::uint32_t), stream),
        "cudaMemset(BAM trace block SM IDs)");

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(begin, stream), "cudaEventRecord");
    const std::size_t threads_per_block = warps_per_block * kWarpSize;
    if (request_mapping == BamTraceRequestMapping::kThreadPerPage) {
        bam_4k_thread_trace_benchmark_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            data_array, device_page_ids.get(), start_index, num_requests,
            page_size_bytes, device_checksum.get(), device_block_sm_ids.get());
        ThrowIfCudaError(
            cudaGetLastError(), "bam_4k_thread_trace_benchmark_kernel");
    } else {
        bam_4k_trace_benchmark_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
            data_array, device_page_ids.get(), start_index, num_requests,
            page_size_bytes, device_checksum.get(), device_block_sm_ids.get());
        ThrowIfCudaError(cudaGetLastError(), "bam_4k_trace_benchmark_kernel");
    }
    if (launch_callback != nullptr) {
        launch_callback(launch_callback_context);
    }
    ThrowIfCudaError(cudaEventRecord(end, stream), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");

    float elapsed_ms = 0.0f;
    ThrowIfCudaError(
        cudaEventElapsedTime(&elapsed_ms, begin, end),
        "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);

    BamTraceBenchmarkResult result;
    result.elapsed_ms = static_cast<double>(elapsed_ms);
    result.block_sm_ids.resize(num_blocks);
    ThrowIfCudaError(
        cudaMemcpyAsync(&result.checksum, device_checksum.get(),
                        sizeof(result.checksum), cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(BAM trace checksum)");
    ThrowIfCudaError(
        cudaMemcpyAsync(result.block_sm_ids.data(), device_block_sm_ids.get(),
                        num_blocks * sizeof(std::uint32_t),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(BAM trace block SM IDs)");
    ThrowIfCudaError(cudaStreamSynchronize(stream),
                     "cudaStreamSynchronize(BAM trace results)");
    return result;
}

BamTraceBenchmarkResult RunBam4kTraceBenchmark(
    const VectorPageProvider& provider,
    const CudaBuffer<std::uint64_t>& device_page_ids,
    std::size_t start_index,
    std::size_t num_requests,
    std::size_t num_blocks,
    std::size_t warps_per_block,
    std::size_t page_size_bytes,
    BamTraceRequestMapping request_mapping) {
    BamTraceBenchmarkWorkspace workspace =
        PrepareBamTraceBenchmarkWorkspace(num_blocks);
    return RunPreparedBam4kTraceBenchmark(
        provider, device_page_ids, start_index, num_requests, num_blocks,
        warps_per_block, &workspace, 0, page_size_bytes, request_mapping);
}


std::vector<std::uint8_t> ReadBamCombinedNodeRecords(
    const VectorPageProvider& provider,
    const std::vector<std::uint32_t>& node_ids,
    std::size_t node_bytes,
    std::size_t nodes_per_page) {
    const auto* data_array =
        static_cast<const array_d_t<char>*>(provider.DeviceReadHandle());
    if (data_array == nullptr) {
        throw std::runtime_error(BuildErrorMessage(
            "ReadBamCombinedNodeRecords",
            "Provider does not expose a BaM device handle."));
    }
    if (node_bytes == 0 || nodes_per_page == 0 ||
        node_bytes * nodes_per_page > kDefaultPageSizeBytes) {
        throw std::runtime_error(BuildErrorMessage(
            "ReadBamCombinedNodeRecords", "Invalid combined-node record layout."));
    }
    if (node_ids.empty()) {
        return {};
    }

    CudaBuffer<std::uint32_t> device_node_ids =
        CudaBuffer<std::uint32_t>::CopyFromHost(node_ids);
    CudaBuffer<std::uint8_t> device_records =
        CudaBuffer<std::uint8_t>::Allocate(node_ids.size() * node_bytes);
    bam_read_combined_node_records_kernel<<<node_ids.size(), kWarpSize>>>(
        data_array, device_node_ids.get(), node_ids.size(), node_bytes,
        nodes_per_page, device_records.get());
    ThrowIfCudaError(cudaGetLastError(),
                     "bam_read_combined_node_records_kernel launch");
    ThrowIfCudaError(cudaDeviceSynchronize(),
                     "bam_read_combined_node_records_kernel synchronize");
    return device_records.CopyToHost();
}

double RunBamFusedExactDistanceFloatQueries(const VectorPageProvider& provider,
                                            ScalarKind vector_scalar_kind,
                                            const CudaBuffer<float>& device_queries,
                                            const CudaBuffer<std::uint64_t>& page_ids,
                                            const CudaBuffer<std::uint32_t>& slot_ids,
                                            const CudaBuffer<std::uint32_t>& node_ids,
                                            const CudaBuffer<std::uint32_t>& candidate_query_ids,
                                            std::size_t num_candidates,
                                            const VectorPageLayout& layout,
                                            std::size_t dim,
                                            CudaBuffer<float>* out_distances) {
    switch (vector_scalar_kind) {
        case ScalarKind::kFloat32:
            return RunBamFusedExactDistanceFloatQueriesImpl<float>(
                provider, device_queries, page_ids, slot_ids, node_ids, candidate_query_ids,
                num_candidates, layout, dim, out_distances);
        case ScalarKind::kUint8:
            return RunBamFusedExactDistanceFloatQueriesImpl<std::uint8_t>(
                provider, device_queries, page_ids, slot_ids, node_ids, candidate_query_ids,
                num_candidates, layout, dim, out_distances);
        case ScalarKind::kInt8:
            return RunBamFusedExactDistanceFloatQueriesImpl<std::int8_t>(
                provider, device_queries, page_ids, slot_ids, node_ids, candidate_query_ids,
                num_candidates, layout, dim, out_distances);
        default:
            throw std::runtime_error(BuildErrorMessage(
                "RunBamFusedExactDistanceFloatQueries", "Unsupported vector scalar kind."));
    }
}

namespace detail {

PersistentBamRerankRunResult RunPersistentBamRerankFloat32(
    const SearchResources& resources,
    const CudaBuffer<float>& device_queries,
    const detail::DeviceTopologyBatchResult& topology_result,
    std::size_t num_queries,
    const RerankExactParams& params) {
    std::vector<float> host_queries;
    if (device_queries.size() != 0) {
        host_queries = device_queries.CopyToHost();
    }
    return RunPersistentBamRerankFloat32(resources, device_queries, host_queries,
                                         topology_result, num_queries, params, 0);
}

PersistentBamRerankRunResult RunPersistentBamRerankFloat32(
    const SearchResources& resources,
    const CudaBuffer<float>& device_queries,
    const std::vector<float>& host_queries,
    const detail::DeviceTopologyBatchResult& topology_result,
    std::size_t num_queries,
    const RerankExactParams& params,
    cudaStream_t stream) {
    const auto& layout = resources.vector_store_layout();
    const bool use_record_layout_override =
        params.ssd_records_per_page != 0 || params.ssd_record_stride_bytes != 0;
    if (use_record_layout_override &&
        (params.ssd_records_per_page == 0 || params.ssd_record_stride_bytes == 0)) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPersistentBamRerankFloat32",
            "SSD record layout override requires both records-per-page and stride."));
    }
    const std::size_t ssd_records_per_page =
        use_record_layout_override ? params.ssd_records_per_page : layout.vectors_per_page();
    const std::size_t ssd_record_stride_bytes =
        use_record_layout_override ? params.ssd_record_stride_bytes : layout.vector_bytes();
    const std::size_t exact_vector_bytes =
        static_cast<std::size_t>(resources.vector_store_header().dim) * sizeof(float);
    if (ssd_record_stride_bytes < exact_vector_bytes ||
        ssd_records_per_page * ssd_record_stride_bytes > layout.page_size_bytes()) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPersistentBamRerankFloat32", "invalid SSD record layout override."));
    }
    const VectorPageProvider& provider = resources.vector_page_provider();
    const auto* data_array = static_cast<const array_d_t<char>*>(provider.DeviceReadHandle());
    if (data_array == nullptr) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPersistentBamRerankFloat32", "Provider does not expose a BaM device handle."));
    }
    if (params.top_k == 0 || params.top_k > params.top_n) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPersistentBamRerankFloat32", "Persistent rerank requires 1 <= top_k <= top_n."));
    }
    if (params.rank_tile_size == 0 || params.rank_tile_size > kMaxPersistentTileSize) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPersistentBamRerankFloat32", "rank_tile_size must be in [1, 128]."));
    }
    if (topology_result.num_queries != num_queries) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPersistentBamRerankFloat32", "topology_result size mismatch."));
    }

    const std::size_t budget_top_n = std::min(params.top_n, topology_result.candidate_capacity);
    const std::string stop_prefix_record_path =
        detail::NonEmptyEnvironmentValue("TOPOANNS_RERANK_STOP_PREFIX_RECORD_PATH");
    const std::string stop_prefix_replay_path =
        detail::NonEmptyEnvironmentValue("TOPOANNS_RERANK_STOP_PREFIX_REPLAY_PATH");
    if (!stop_prefix_record_path.empty() && !stop_prefix_replay_path.empty()) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPersistentBamRerankFloat32", "Record and replay modes are mutually exclusive."));
    }
    if ((!stop_prefix_record_path.empty() || !stop_prefix_replay_path.empty()) &&
        !params.use_learned_stop) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPersistentBamRerankFloat32", "Stop-prefix profiling requires learned stop."));
    }
    const std::size_t result_top_n = params.use_learned_stop ? params.top_k : budget_top_n;
    detail::PersistentBamRerankRunResult result;
    result.result_top_n = result_top_n;
    result.final_distances = CudaBuffer<float>::Allocate(num_queries * result_top_n);
    result.final_node_ids = CudaBuffer<std::uint32_t>::Allocate(num_queries * result_top_n);
    CudaBuffer<std::uint32_t> exact_counts = CudaBuffer<std::uint32_t>::Allocate(num_queries);
    CudaBuffer<std::uint32_t> rerank_ssd_io_counts =
        CudaBuffer<std::uint32_t>::Allocate(num_queries);
    CudaBuffer<std::uint32_t> reused_exact_counts =
        CudaBuffer<std::uint32_t>::Allocate(num_queries);
    CudaBuffer<std::uint64_t> exact_reuse_lookup_cycles =
        CudaBuffer<std::uint64_t>::Allocate(num_queries);
    CudaBuffer<std::uint32_t> filtered_counts = CudaBuffer<std::uint32_t>::Allocate(num_queries);

    DeviceRerankLearnedStopConfig learned_stop{};
    bool use_learned_schedule = params.use_learned_stop;
    if (use_learned_schedule) {
        learned_stop = LoadRerankLearnedStopConfigFromFile(
            params.learned_stop_model_path, static_cast<std::uint32_t>(budget_top_n),
            static_cast<std::uint32_t>(params.top_k));
        use_learned_schedule = learned_stop.enabled != 0U;
    }
    const bool evaluate_learned_stop =
        use_learned_schedule && stop_prefix_replay_path.empty();
    CudaBuffer<std::uint32_t> replay_stop_prefixes;
    if (!stop_prefix_replay_path.empty()) {
        std::vector<std::uint32_t> host_replay_stop_prefixes =
            detail::ReadRerankStopPrefixTrace(
                stop_prefix_replay_path, budget_top_n, num_queries);
        for (std::uint32_t prefix : host_replay_stop_prefixes) {
            if (prefix < params.top_k || prefix > budget_top_n) {
                throw std::runtime_error(BuildErrorMessage(
                    "RunPersistentBamRerankFloat32", "Invalid replay stop prefix."));
            }
        }
        replay_stop_prefixes =
            CudaBuffer<std::uint32_t>::CopyFromHost(host_replay_stop_prefixes);
    }
    ThrowIfCudaError(cudaMemcpyToSymbol(kDeviceRerankLearnedStopConfig, &learned_stop,
                                        sizeof(DeviceRerankLearnedStopConfig)),
                     "cudaMemcpyToSymbol(kDeviceRerankLearnedStopConfig)");

    CudaBuffer<float> query_norm_squares;
    CudaBuffer<float> lower_bounds;
    double lower_bound_kernel_ms = 0.0;
    if (params.use_pq2_bound_filter) {
        std::vector<float> host_query_norm_squares(num_queries, 0.0f);
        const std::size_t dim = resources.vector_store_header().dim;
        if (host_queries.size() != num_queries * dim) {
            throw std::runtime_error(BuildErrorMessage(
                "RunPersistentBamRerankFloat32",
                "host query buffer size must equal num_queries * dim for stream-aware rerank."));
        }
        for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
            const float* query_ptr = host_queries.data() + query_id * dim;
            double sum = 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                const double value = static_cast<double>(query_ptr[d]);
                sum += value * value;
            }
            host_query_norm_squares[query_id] = static_cast<float>(sum);
        }
        query_norm_squares = CudaBuffer<float>::Allocate(host_query_norm_squares.size());
        ThrowIfCudaError(cudaMemcpyAsync(query_norm_squares.get(), host_query_norm_squares.data(),
                                         host_query_norm_squares.size() * sizeof(float),
                                         cudaMemcpyHostToDevice, stream),
                         "cudaMemcpyAsyncHostToDevice");
        lower_bounds =
            CudaBuffer<float>::Allocate(num_queries * topology_result.candidate_capacity);
        lower_bound_kernel_ms = RecordPrecomputeLowerBounds(
            topology_result.candidate_buffer.get(), topology_result.candidate_capacity,
            budget_top_n, num_queries, resources.pq2_error_bounds_fp32().get(),
            query_norm_squares.get(), lower_bounds.get(), stream);
    }

    const bool enable_learned_fine_profile =
        evaluate_learned_stop && RerankLearnedFineProfileEnabled();
    CudaBuffer<std::uint64_t> learned_cycles;
    CudaBuffer<std::uint64_t> checkpoint_bookkeeping_cycles;
    CudaBuffer<std::uint64_t> topk_churn_cycles;
    CudaBuffer<std::uint64_t> next_window_scan_cycles;
    CudaBuffer<std::uint64_t> logit_eval_cycles;
    CudaBuffer<std::uint64_t> query_block_cycles =
        CudaBuffer<std::uint64_t>::Allocate(num_queries);
    CudaBuffer<std::uint32_t> checkpoint_counts;
    CudaBuffer<std::uint32_t> stop_prefixes;
    CudaBuffer<std::uint32_t> stop_flags;
    if (evaluate_learned_stop) {
        learned_cycles = CudaBuffer<std::uint64_t>::Allocate(num_queries);
        checkpoint_bookkeeping_cycles = CudaBuffer<std::uint64_t>::Allocate(num_queries);
        topk_churn_cycles = CudaBuffer<std::uint64_t>::Allocate(num_queries);
        next_window_scan_cycles = CudaBuffer<std::uint64_t>::Allocate(num_queries);
        logit_eval_cycles = CudaBuffer<std::uint64_t>::Allocate(num_queries);
        checkpoint_counts = CudaBuffer<std::uint32_t>::Allocate(num_queries);
        stop_prefixes = CudaBuffer<std::uint32_t>::Allocate(num_queries);
        stop_flags = CudaBuffer<std::uint32_t>::Allocate(num_queries);
    }

    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin, stream), "cudaEventRecord");
    const std::size_t dim = resources.vector_store_header().dim;
    if (params.use_pq2_bound_filter) {
        LaunchPersistentBamRerankKernelFloat32Specialized<true>(
            dim, num_queries, stream, data_array, device_queries.get(),
            topology_result.candidate_buffer.get(), topology_result.candidate_capacity,
            budget_top_n, result_top_n, params.rank_tile_size, ssd_records_per_page,
            layout.page_size_bytes(), ssd_record_stride_bytes, params.top_k,
            params.use_early_stop, params.early_stop_min_prefix,
            params.early_stop_patience_tiles, use_learned_schedule, evaluate_learned_stop,
            enable_learned_fine_profile, replay_stop_prefixes.get(), resources.pq2_error_bounds_fp32().get(),
            query_norm_squares.get(), lower_bounds.get(),
            topology_result.exact_reuse_node_ids.empty()
                ? nullptr
                : topology_result.exact_reuse_node_ids.get(),
            topology_result.exact_reuse_distances.empty()
                ? nullptr
                : topology_result.exact_reuse_distances.get(),
            topology_result.exact_reuse_cache_capacity, result.final_distances.get(),
            result.final_node_ids.get(), exact_counts.get(), rerank_ssd_io_counts.get(),
            reused_exact_counts.get(), filtered_counts.get(), learned_cycles.get(),
            checkpoint_bookkeeping_cycles.get(), topk_churn_cycles.get(),
            next_window_scan_cycles.get(), logit_eval_cycles.get(),
            exact_reuse_lookup_cycles.get(), query_block_cycles.get(), checkpoint_counts.get(), stop_prefixes.get(),
            stop_flags.get());
    } else {
        LaunchPersistentBamRerankKernelFloat32Specialized<false>(
            dim, num_queries, stream, data_array, device_queries.get(),
            topology_result.candidate_buffer.get(), topology_result.candidate_capacity,
            budget_top_n, result_top_n, params.rank_tile_size, ssd_records_per_page,
            layout.page_size_bytes(), ssd_record_stride_bytes, params.top_k,
            params.use_early_stop, params.early_stop_min_prefix,
            params.early_stop_patience_tiles, false, false, false, nullptr, nullptr, nullptr, nullptr,
            topology_result.exact_reuse_node_ids.empty()
                ? nullptr
                : topology_result.exact_reuse_node_ids.get(),
            topology_result.exact_reuse_distances.empty()
                ? nullptr
                : topology_result.exact_reuse_distances.get(),
            topology_result.exact_reuse_cache_capacity, result.final_distances.get(),
            result.final_node_ids.get(), exact_counts.get(), rerank_ssd_io_counts.get(),
            reused_exact_counts.get(), filtered_counts.get(), nullptr, nullptr, nullptr,
            nullptr, nullptr, exact_reuse_lookup_cycles.get(), query_block_cycles.get(), nullptr, nullptr, nullptr);
    }
    ThrowIfCudaError(cudaGetLastError(), "persistent_bam_rerank_kernel_float32_dim");
    ThrowIfCudaError(cudaEventRecord(kernel_end, stream), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);
    result.kernel_ms = static_cast<double>(kernel_ms) + lower_bound_kernel_ms;

    const std::vector<std::uint32_t> host_exact_counts = exact_counts.CopyToHost();
    const std::vector<std::uint32_t> host_rerank_ssd_io_counts =
        rerank_ssd_io_counts.CopyToHost();
    const std::vector<std::uint32_t> host_reused_exact_counts =
        reused_exact_counts.CopyToHost();
    const std::vector<std::uint32_t> host_filtered_counts = filtered_counts.CopyToHost();
    const std::vector<std::uint64_t> host_exact_reuse_lookup_cycles =
        exact_reuse_lookup_cycles.CopyToHost();
    const std::vector<std::uint64_t> host_query_block_cycles =
        query_block_cycles.CopyToHost();
    std::uint64_t total_exact_reuse_lookup_cycles = 0;
    std::uint64_t total_query_block_cycles = 0;
    for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
        result.exact_count += host_exact_counts[query_id];
        result.rerank_ssd_io_pages += host_rerank_ssd_io_counts[query_id];
        result.reused_exact_count += host_reused_exact_counts[query_id];
        result.bound_filtered_count += host_filtered_counts[query_id];
        total_exact_reuse_lookup_cycles += host_exact_reuse_lookup_cycles[query_id];
        total_query_block_cycles += host_query_block_cycles[query_id];
    }
    result.query_block_ms = CyclesToMilliseconds(total_query_block_cycles);
    result.exact_reuse_lookup_ms =
        CyclesToMilliseconds(total_exact_reuse_lookup_cycles);

    if (evaluate_learned_stop) {
        const std::vector<std::uint64_t> host_learned_cycles = learned_cycles.CopyToHost();
        const std::vector<std::uint64_t> host_checkpoint_bookkeeping_cycles =
            checkpoint_bookkeeping_cycles.CopyToHost();
        const std::vector<std::uint64_t> host_topk_churn_cycles =
            topk_churn_cycles.CopyToHost();
        const std::vector<std::uint64_t> host_next_window_scan_cycles =
            next_window_scan_cycles.CopyToHost();
        const std::vector<std::uint64_t> host_logit_eval_cycles =
            logit_eval_cycles.CopyToHost();
        const std::vector<std::uint32_t> host_checkpoint_counts =
            checkpoint_counts.CopyToHost();
        const std::vector<std::uint32_t> host_stop_prefixes = stop_prefixes.CopyToHost();
        if (!stop_prefix_record_path.empty()) {
            detail::AppendRerankStopPrefixTrace(
                stop_prefix_record_path, budget_top_n, host_stop_prefixes);
        }
        std::uint64_t learned_sum = 0;
        std::uint64_t checkpoint_bookkeeping_sum = 0;
        std::uint64_t topk_churn_sum = 0;
        std::uint64_t next_window_scan_sum = 0;
        std::uint64_t logit_eval_sum = 0;
        result.learned_stop_queries = num_queries;
        result.learned_stop_prefix_min = budget_top_n;
        result.learned_stop_prefix_max = 0;
        for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
            learned_sum += host_learned_cycles[query_id];
            checkpoint_bookkeeping_sum += host_checkpoint_bookkeeping_cycles[query_id];
            topk_churn_sum += host_topk_churn_cycles[query_id];
            next_window_scan_sum += host_next_window_scan_cycles[query_id];
            logit_eval_sum += host_logit_eval_cycles[query_id];
            result.learned_stop_checkpoints += host_checkpoint_counts[query_id];
            const std::size_t prefix = host_stop_prefixes[query_id];
            result.learned_stop_prefix_sum += prefix;
            result.learned_stop_prefix_min =
                std::min(result.learned_stop_prefix_min, prefix);
            result.learned_stop_prefix_max =
                std::max(result.learned_stop_prefix_max, prefix);
        }
        result.learned_stop_model_ms = CyclesToMilliseconds(learned_sum);
        result.learned_stop_checkpoint_bookkeeping_ms =
            CyclesToMilliseconds(checkpoint_bookkeeping_sum);
        result.learned_stop_topk_churn_ms = CyclesToMilliseconds(topk_churn_sum);
        result.learned_stop_next_window_scan_ms =
            CyclesToMilliseconds(next_window_scan_sum);
        result.learned_stop_logit_eval_ms = CyclesToMilliseconds(logit_eval_sum);
    }
    return result;
}

}  // namespace detail

void WriteVectorStorePayloadToBam(const std::filesystem::path& vector_store_path,
                                  std::size_t header_bytes,
                                  std::size_t page_size_bytes,
                                  const BamVectorProviderOptions& options) {
    WriteFilePayloadToBam(vector_store_path, header_bytes, page_size_bytes, options);
}

void WriteFilePayloadToBam(const std::filesystem::path& path,
                           std::size_t header_bytes,
                           std::size_t page_size_bytes,
                           const BamVectorProviderOptions& options) {
    ValidateBamLayout(path, header_bytes, page_size_bytes, options.device_offset_bytes,
                      options.payload_bytes_override);
    const std::uint64_t file_size = CheckedFileSize(path);
    const std::uint64_t file_payload_bytes = file_size - header_bytes;
    const std::uint64_t effective_payload_bytes =
        ResolveEffectivePayloadBytes(path, header_bytes, options);

    Settings settings = MakeBamSettings(options, page_size_bytes);
    const std::string controller_path_string = options.controller_path.string();
    std::vector<std::unique_ptr<Controller>> controllers;
    try {
        controllers.push_back(std::make_unique<Controller>(
            controller_path_string.c_str(), settings.nvmNamespace, settings.cudaDevice,
            settings.queueDepth, settings.numQueues));
    } catch (const std::exception& e) {
        throw BuildBamInitError("WriteFilePayloadToBam", path, options.cuda_device, e);
    }
    std::vector<Controller*> controller_ptrs = {controllers.front().get()};

    const std::uint64_t cache_pages =
        std::max<std::uint64_t>(1, settings.maxPageCacheSize / settings.pageSize);
    page_cache_t host_page_cache(
        settings.pageSize, cache_pages, settings.cudaDevice,
        *controller_ptrs.front(), options.max_ranges, controller_ptrs);
    auto* device_page_cache = static_cast<page_cache_d_t*>(host_page_cache.d_pc_ptr);
    settings.numPages = settings.maxPageCacheSize / settings.pageSize;
    const std::uint64_t chunk_bytes =
        static_cast<std::uint64_t>(settings.numPages) * settings.pageSize;
    if (chunk_bytes == 0) {
        throw std::runtime_error(BuildErrorMessage("WriteFilePayloadToBam",
                                                   "BAM page cache is smaller than one page."));
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("WriteFilePayloadToBam", "Failed to open " + path.string()));
    }
    in.seekg(static_cast<std::streamoff>(header_bytes), std::ios::beg);
    if (!in.good()) {
        throw std::runtime_error(BuildErrorMessage("WriteFilePayloadToBam",
                                                   "Failed to seek past file header."));
    }

    std::vector<char> host_buffer(static_cast<std::size_t>(chunk_bytes), 0);
    std::uint64_t written_bytes = 0;
    std::cout << "[topoanns_bam_write_payload]"
              << " path=" << path
              << " payload_bytes=" << file_payload_bytes
              << " effective_payload_bytes=" << effective_payload_bytes
              << " device_offset_bytes=" << options.device_offset_bytes
              << " chunk_bytes=" << chunk_bytes
              << std::endl;

    while (written_bytes < effective_payload_bytes) {
        const std::uint64_t logical_chunk_bytes =
            std::min<std::uint64_t>(chunk_bytes, effective_payload_bytes - written_bytes);
        std::fill(host_buffer.begin(), host_buffer.end(), 0);

        const std::uint64_t remaining_file_bytes =
            written_bytes < file_payload_bytes ? file_payload_bytes - written_bytes : 0;
        const std::uint64_t read_bytes =
            std::min<std::uint64_t>(logical_chunk_bytes, remaining_file_bytes);
        if (read_bytes != 0) {
            in.read(host_buffer.data(), static_cast<std::streamsize>(read_bytes));
            if (in.gcount() != static_cast<std::streamsize>(read_bytes)) {
                throw std::runtime_error(BuildErrorMessage("WriteFilePayloadToBam",
                                                           "Short read while writing payload."));
            }
        }

        cuda_err_chk(cudaMemcpy(host_page_cache.pdt.base_addr,
                                host_buffer.data(),
                                static_cast<std::size_t>(chunk_bytes),
                                cudaMemcpyHostToDevice));

        const std::uint64_t block_size = settings.blkSize;
        const std::uint64_t grid_size = (settings.numThreads + block_size - 1) / block_size;
        sequential_access_kernel<<<grid_size, block_size>>>(
            host_page_cache.pdt.d_ctrls, device_page_cache, settings.pageSize,
            settings.numThreads, settings.n_ctrls, settings.numReqs, WRITE,
            options.device_offset_bytes + written_bytes, 0);
        cuda_err_chk(cudaDeviceSynchronize());

        written_bytes += logical_chunk_bytes;
        std::cout << "[topoanns_bam_write_payload] wrote_logical_bytes=" << written_bytes
                  << " remaining_logical_bytes="
                  << (effective_payload_bytes - written_bytes) << std::endl;
    }
}

}  // namespace topoanns
