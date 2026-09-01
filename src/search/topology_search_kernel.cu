#include "topology_search_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "topoanns/cuda_buffer.hpp"
#include "topology_microbatch_io.hpp"

namespace topoanns::detail {

__device__ void CopyTopologyNeighborsFromBam(const void* topology_ssd_handle,
                                             std::size_t page_offset,
                                             std::size_t node_offset_in_page,
                                             std::uint32_t degree,
                                             std::uint32_t* shared_dst,
                                             std::uint32_t lane_id,
                                             std::uint32_t active_mask);

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
                                         std::uint32_t active_mask);

namespace {

constexpr std::uint32_t kInvalidFullPrefixIteration = 0xffffffffU;

constexpr unsigned int kFullMask = 0xffffffffU;
constexpr std::size_t kVisitedHashWords = 1024;
constexpr std::size_t kVisitedHashFunctions = 3;
constexpr std::size_t kWarpSize = 32;
constexpr std::size_t kMaxSearchWidth = 32;
constexpr std::size_t kTopologyBlockThreads = 128;
constexpr std::size_t kNoFixedNumChunks = 0;
constexpr std::size_t kSpecializedSearchPqNumChunks = 34;
// The runtime shared-memory check below is the authoritative device-specific limit.
constexpr std::size_t kMaxFrontierPqTileChunks = 48;
constexpr std::size_t kLearnedStopFeatureCount = 12;
constexpr std::size_t kMaxLearnedStopStages = 32;
constexpr std::size_t kMaxLearnedStopTopK = 32;
constexpr std::uint32_t kTopkChurnModeRatio = 0U;
constexpr std::uint32_t kTopkChurnModeFlag = 1U;
constexpr std::uint32_t kNext64ModeFrac = 0U;
constexpr std::uint32_t kNext64ModeFlag = 1U;
constexpr std::uint32_t kFrontierPqModeCurrent = 0U;
constexpr std::uint32_t kFrontierPqModeBang = 1U;
constexpr std::uint32_t kFrontierPqModeGustann = 2U;
constexpr std::uint32_t kTraversalLearnedStopFeatureVariantGeneric = 0U;
constexpr std::uint32_t kTraversalLearnedStopFeatureVariantCore7 = 1U;
constexpr std::array<std::uint32_t, kLearnedStopFeatureCount>
    kTraversalLearnedStopCore7FeatureMask = {0U, 0U, 1U, 1U, 0U, 0U,
                                             1U, 1U, 1U, 0U, 1U, 1U};
using vectype = ulonglong4;

using DeviceCandidate = DeviceTopologyCandidate;
using DeviceSearchStats = DeviceTopologySearchStats;
using DeviceProfileCycles = DeviceTopologyProfileCycles;

struct DeviceTraversalExactReuseConfig {
    std::uint32_t enabled = 0;
    std::uint32_t query_dim = 0;
    std::uint32_t vector_bytes = 0;
    std::uint32_t node_bytes = 0;
    std::uint32_t nodes_per_page = 0;
    std::uint32_t cache_capacity = 0;
    const void* combined_ssd = nullptr;
    const float* queries = nullptr;
    std::uint32_t* cache_node_ids = nullptr;
    float* cache_distances = nullptr;
};

struct DeviceLearnedStopConfig {
    std::uint32_t enabled = 0U;
    std::uint32_t feature_variant = kTraversalLearnedStopFeatureVariantGeneric;
    std::uint32_t num_stages = 0U;
    std::uint32_t top_k = 0U;
    std::uint32_t stage_prefixes[kMaxLearnedStopStages] = {};
    float linear_weights[kLearnedStopFeatureCount] = {};
    float bias = 0.0f;
    float threshold_logit = 0.0f;
    float inv_top_l = 0.0f;
    std::uint32_t shadow_mode = 0U;
    std::uint32_t topk_churn_mode = kTopkChurnModeRatio;
    std::uint32_t next64_mode = kNext64ModeFrac;
    std::uint32_t feature_mask[kLearnedStopFeatureCount] = {};
};

__constant__ DeviceLearnedStopConfig kDeviceTraversalLearnedStopConfig;

struct DeviceMicrobatchQueryState {
    std::uint32_t active = 0U;
    std::uint32_t hash_iteration = 0U;
    std::uint32_t merge_ordinal = 0U;
    std::uint32_t learned_stop_stage = 0U;
    std::uint32_t learned_stop_action = 0U;
    std::uint32_t learned_stop_has_prev = 0U;
    std::uint32_t learned_stop_topk_changed = 0U;
    std::uint32_t selected_count = 0U;
    std::uint32_t selected_uncached_mask = 0U;
    float learned_stop_prev_dk = std::numeric_limits<float>::infinity();
    float learned_stop_prev_boundary = std::numeric_limits<float>::infinity();
    std::uint32_t learned_stop_prev_topk_ids[kMaxLearnedStopTopK] = {};
};

struct ParsedLearnedStopModel {
    std::vector<std::uint32_t> stage_prefixes;
    std::vector<float> feature_means;
    std::vector<float> feature_inv_stds;
    std::vector<float> weights;
    float bias = 0.0f;
    float threshold_logit = 0.0f;
    std::uint32_t file_top_k = 0U;
    std::string topk_churn_mode = "ratio";
    std::string next64_mode = "frac";
    std::vector<std::uint32_t> feature_mask;
    std::unordered_map<std::uint32_t, float> per_top_l_threshold_logits;
};

__host__ inline std::string TrimAscii(std::string text) {
    const auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    text.erase(text.begin(), std::find_if(text.begin(), text.end(), not_space));
    text.erase(std::find_if(text.rbegin(), text.rend(), not_space).base(), text.end());
    return text;
}

__host__ std::vector<std::string> SplitCsvTokens(const std::string& text) {
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

__host__ std::vector<std::uint32_t> ParseCsvU32(const std::string& text) {
    std::vector<std::uint32_t> values;
    for (const std::string& token : SplitCsvTokens(text)) {
        values.push_back(static_cast<std::uint32_t>(std::stoul(token)));
    }
    return values;
}

__host__ std::vector<float> ParseCsvFloat(const std::string& text) {
    std::vector<float> values;
    for (const std::string& token : SplitCsvTokens(text)) {
        values.push_back(std::stof(token));
    }
    return values;
}

template <typename KernelT>
__host__ void ConfigureKernelDynamicSharedMemory(KernelT kernel,
                                                 std::size_t dynamic_shared_bytes) {
    constexpr std::size_t kDefaultSharedMemoryLimit = 48 * 1024;
    if (dynamic_shared_bytes <= kDefaultSharedMemoryLimit) {
        return;
    }
    ThrowIfCudaError(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(dynamic_shared_bytes)),
        "cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)");
    ThrowIfCudaError(
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100),
        "cudaFuncSetAttribute(cudaFuncAttributePreferredSharedMemoryCarveout)");
}

template <typename KernelT>
__host__ void PopulateKernelOccupancy(KernelT kernel,
                                      std::size_t dynamic_shared_bytes,
                                      std::size_t search_width,
                                      std::size_t num_queries,
                                      bool enabled,
                                      DeviceTopologyBatchResult* result) {
    if (!enabled || result == nullptr) {
        return;
    }
    int device = 0;
    ThrowIfCudaError(cudaGetDevice(&device), "cudaGetDevice(occupancy)");
    const std::uint64_t cache_key =
        (static_cast<std::uint64_t>(static_cast<std::uint32_t>(device)) << 56U) |
        static_cast<std::uint64_t>(dynamic_shared_bytes);
    static thread_local std::unordered_map<std::uint64_t, std::pair<int, int>> cache;
    auto cached = cache.find(cache_key);
    if (cached == cache.end()) {
        int blocks_per_sm = 0;
        ThrowIfCudaError(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &blocks_per_sm, kernel, static_cast<int>(kTopologyBlockThreads),
                dynamic_shared_bytes),
            "cudaOccupancyMaxActiveBlocksPerMultiprocessor(topology)");
        int sm_count = 0;
        ThrowIfCudaError(
            cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device),
            "cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)");
        cached = cache.emplace(cache_key, std::make_pair(blocks_per_sm, sm_count)).first;
    }
    const int blocks_per_sm = cached->second.first;
    const int sm_count = cached->second.second;
    const std::size_t resident_capacity =
        static_cast<std::size_t>(std::max(0, blocks_per_sm)) *
        static_cast<std::size_t>(std::max(0, sm_count));
    result->occupancy_dynamic_shared_bytes = dynamic_shared_bytes;
    result->occupancy_blocks_per_sm =
        static_cast<std::size_t>(std::max(0, blocks_per_sm));
    result->occupancy_sm_count = static_cast<std::size_t>(std::max(0, sm_count));
    result->occupancy_resident_blocks = std::min(resident_capacity, num_queries);
    result->occupancy_max_io_warps =
        result->occupancy_resident_blocks *
        std::min(search_width, kTopologyBlockThreads / kWarpSize);
}

__host__ ParsedLearnedStopModel ParseLearnedStopModelFile(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error(BuildErrorMessage(
            "LoadLearnedStopConfigFromFile", "Failed to open " + path.string()));
    }

    ParsedLearnedStopModel model;

    std::string line;
    while (std::getline(in, line)) {
        line = TrimAscii(std::move(line));
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const std::size_t eq = line.find('=');
        if (eq == std::string::npos) {
            throw std::runtime_error(BuildErrorMessage(
                "LoadLearnedStopConfigFromFile",
                "Invalid learned-stop config line: " + line));
        }
        const std::string key = TrimAscii(line.substr(0, eq));
        const std::string value = TrimAscii(line.substr(eq + 1));
        if (key == "stage_prefixes") {
            model.stage_prefixes = ParseCsvU32(value);
        } else if (key == "feature_means") {
            model.feature_means = ParseCsvFloat(value);
        } else if (key == "feature_inv_stds") {
            model.feature_inv_stds = ParseCsvFloat(value);
        } else if (key == "weights") {
            model.weights = ParseCsvFloat(value);
        } else if (key == "bias") {
            model.bias = std::stof(value);
        } else if (key == "threshold_logit") {
            model.threshold_logit = std::stof(value);
        } else if (key == "top_k") {
            model.file_top_k = static_cast<std::uint32_t>(std::stoul(value));
        } else if (key == "topk_churn_mode") {
            model.topk_churn_mode = value;
        } else if (key == "next64_mode") {
            model.next64_mode = value;
        } else if (key == "feature_mask") {
            model.feature_mask = ParseCsvU32(value);
        } else if (key.rfind("threshold_logit_top_l_", 0) == 0U) {
            const std::string suffix =
                key.substr(std::char_traits<char>::length("threshold_logit_top_l_"));
            if (!suffix.empty()) {
                model.per_top_l_threshold_logits.emplace(
                    static_cast<std::uint32_t>(std::stoul(suffix)), std::stof(value));
            }
        }
    }

    if (model.feature_means.size() != kLearnedStopFeatureCount ||
        model.feature_inv_stds.size() != kLearnedStopFeatureCount ||
        model.weights.size() != kLearnedStopFeatureCount) {
        throw std::runtime_error(BuildErrorMessage(
            "LoadLearnedStopConfigFromFile",
            "learned-stop feature vectors must all have length " +
                std::to_string(kLearnedStopFeatureCount) + "."));
    }

    return model;
}

__host__ const ParsedLearnedStopModel& GetParsedLearnedStopModel(
    const std::filesystem::path& path) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, ParsedLearnedStopModel> cache;
    const std::string key = path.lexically_normal().string();
    std::lock_guard<std::mutex> lock(cache_mutex);
    const auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    auto [inserted_it, _] = cache.emplace(key, ParseLearnedStopModelFile(path));
    return inserted_it->second;
}

__host__ DeviceLearnedStopConfig LoadLearnedStopConfigFromFile(
    const std::filesystem::path& path,
    std::uint32_t top_l,
    std::uint32_t top_k) {
    const ParsedLearnedStopModel& model = GetParsedLearnedStopModel(path);
    DeviceLearnedStopConfig config{};
    config.top_k = model.file_top_k != 0U ? model.file_top_k : top_k;
    config.inv_top_l = top_l == 0U ? 0.0f : 1.0f / static_cast<float>(top_l);
    if (const char* env = std::getenv("TOPOANNS_LEARNED_STOP_SHADOW")) {
        const std::string value = TrimAscii(env);
        if (!(value.empty() || value == "0" || value == "false" ||
              value == "FALSE")) {
            config.shadow_mode = 1U;
        }
    }
    config.topk_churn_mode =
        model.topk_churn_mode == "flag" ? kTopkChurnModeFlag : kTopkChurnModeRatio;
    config.next64_mode =
        model.next64_mode == "flag" ? kNext64ModeFlag : kNext64ModeFrac;
    const auto per_top_l_it = model.per_top_l_threshold_logits.find(top_l);
    config.threshold_logit =
        per_top_l_it != model.per_top_l_threshold_logits.end() ? per_top_l_it->second
                                                               : model.threshold_logit;
    float linear_bias = model.bias;
    bool matches_core7 = config.topk_churn_mode == kTopkChurnModeFlag;
    for (std::size_t i = 0; i < kLearnedStopFeatureCount; ++i) {
        config.feature_mask[i] =
            i < model.feature_mask.size() ? model.feature_mask[i] : 1U;
        const float masked_weight =
            config.feature_mask[i] != 0U ? model.weights[i] : 0.0f;
        config.linear_weights[i] = masked_weight * model.feature_inv_stds[i];
        linear_bias -= model.feature_means[i] * config.linear_weights[i];
        matches_core7 =
            matches_core7 && config.feature_mask[i] == kTraversalLearnedStopCore7FeatureMask[i];
    }
    config.bias = linear_bias;
    if (matches_core7) {
        config.feature_variant = kTraversalLearnedStopFeatureVariantCore7;
    }
    for (std::uint32_t prefix : model.stage_prefixes) {
        if (prefix == 0U || prefix >= top_l) {
            continue;
        }
        if (config.num_stages >= kMaxLearnedStopStages) {
            break;
        }
        config.stage_prefixes[config.num_stages++] = prefix;
    }
    config.enabled = config.num_stages == 0U ? 0U : 1U;
    if (config.top_k == 0U || config.top_k > kMaxLearnedStopTopK) {
        config.enabled = 0U;
    }
    return config;
}

__device__ __constant__ std::uint32_t kVisitedHashSeeds[10] = {
    0x924ed183U, 0xd854fc0aU, 0xecf5e3b7U, 0x1bead407U, 0x28a30449U,
    0xbfc4d99fU, 0x715030e2U, 0xffcfb45bU, 0x6e4ce166U, 0xeb53c362U,
};

__host__ __device__ inline std::size_t NextPowerOfTwo(std::size_t value) {
    std::size_t power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

__host__ __device__ inline DeviceCandidate InvalidDeviceCandidate() {
    DeviceCandidate candidate;
    candidate.distance = std::numeric_limits<float>::infinity();
    candidate.node_id = kInvalidNodeId;
    return candidate;
}

template <typename T>
__host__ __device__ inline T* AlignPointer(void* pointer) {
    auto address = reinterpret_cast<std::uintptr_t>(pointer);
    constexpr std::uintptr_t alignment = alignof(T);
    address = (address + alignment - 1U) & ~(alignment - 1U);
    return reinterpret_cast<T*>(address);
}

__host__ __device__ __forceinline__ bool IsValid(const DeviceCandidate& candidate) {
    return candidate.valid();
}

__device__ __forceinline__ bool CandidateLess(const DeviceCandidate& lhs,
                                              const DeviceCandidate& rhs) {
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    return lhs.raw_node_id() < rhs.raw_node_id();
}

__device__ __forceinline__ std::uint32_t UpperBoundCandidate(const DeviceCandidate* candidates,
                                                             std::uint32_t count,
                                                             const DeviceCandidate& value) {
    std::uint32_t begin = 0U;
    std::uint32_t end = count;
    while (begin < end) {
        const std::uint32_t mid = (begin + end) >> 1U;
        if (!CandidateLess(value, candidates[mid])) {
            begin = mid + 1U;
        } else {
            end = mid;
        }
    }
    return begin;
}

__device__ __forceinline__ std::uint32_t get_oid(std::uint32_t eq_mask) {
    const std::uint32_t lane = static_cast<std::uint32_t>(threadIdx.x) & (kWarpSize - 1U);
    const std::uint32_t lower_mask = lane == 0U ? 0U : ((1U << lane) - 1U);
    return __popc(eq_mask & lower_mask);
}

__device__ __forceinline__ float WarpReduceSum(float value) {
    value += __shfl_down_sync(kFullMask, value, 16);
    value += __shfl_down_sync(kFullMask, value, 8);
    value += __shfl_down_sync(kFullMask, value, 4);
    value += __shfl_down_sync(kFullMask, value, 2);
    value += __shfl_down_sync(kFullMask, value, 1);
    return value;
}

__device__ __forceinline__ float GroupReduceSumWidth8(float value) {
    value += __shfl_down_sync(kFullMask, value, 4, 8);
    value += __shfl_down_sync(kFullMask, value, 2, 8);
    value += __shfl_down_sync(kFullMask, value, 1, 8);
    return value;
}

__device__ __forceinline__ std::uint64_t MixChecksum(std::uint64_t seed, std::uint64_t value) {
    seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
    return seed;
}

__device__ std::uint64_t ChecksumCandidates(const DeviceCandidate* candidates, std::size_t count) {
    std::uint64_t checksum = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < count; ++i) {
        checksum = MixChecksum(checksum, static_cast<std::uint64_t>(i));
        checksum = MixChecksum(checksum, static_cast<std::uint64_t>(candidates[i].raw_node_id()));
        checksum = MixChecksum(
            checksum, static_cast<std::uint64_t>(__float_as_uint(candidates[i].distance)));
        checksum = MixChecksum(checksum, static_cast<std::uint64_t>(candidates[i].expanded()));
    }
    return checksum;
}

__device__ std::uint64_t ChecksumWords(const std::uint32_t* words, std::size_t count) {
    std::uint64_t checksum = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < count; ++i) {
        checksum = MixChecksum(checksum, static_cast<std::uint64_t>(i));
        checksum = MixChecksum(checksum, static_cast<std::uint64_t>(words[i]));
    }
    return checksum;
}

__forceinline__ __device__ int copyFromHostCacheVec(const void* data_in,
                                                    void* data_out,
                                                    std::size_t bytes,
                                                    std::uint32_t eq_mask) {
    const std::uint32_t num_threads = __popc(eq_mask);
    const std::uint32_t oid = get_oid(eq_mask);
    const std::uintptr_t combined_alignment =
        reinterpret_cast<std::uintptr_t>(data_in) |
        reinterpret_cast<std::uintptr_t>(data_out) |
        static_cast<std::uintptr_t>(bytes);

    if ((combined_alignment & (alignof(vectype) - 1U)) == 0U) {
        const auto* data_in_vec = reinterpret_cast<const vectype*>(data_in);
        auto* data_out_vec = reinterpret_cast<vectype*>(data_out);
        const std::size_t vec_count = bytes / sizeof(vectype);
        for (std::size_t i = oid; i < vec_count; i += num_threads) {
            data_out_vec[i] = data_in_vec[i];
        }
        return static_cast<int>(bytes);
    }

    const auto* data_in_byte = reinterpret_cast<const std::uint8_t*>(data_in);
    auto* data_out_byte = reinterpret_cast<std::uint8_t*>(data_out);
    for (std::size_t i = oid; i < bytes; i += num_threads) {
        data_out_byte[i] = data_in_byte[i];
    }
    return static_cast<int>(bytes);
}

__device__ __forceinline__ std::uint32_t HashExactReuseNode(std::uint32_t value) {
    value ^= value >> 16;
    value *= 0x7feb352dU;
    value ^= value >> 15;
    value *= 0x846ca68bU;
    value ^= value >> 16;
    return value;
}

__device__ __forceinline__ void InsertExactReuse(
    const DeviceTraversalExactReuseConfig& exact_reuse,
    std::size_t query_id,
    std::uint32_t node_id,
    float distance,
    DeviceSearchStats* stats) {
    std::uint32_t* node_ids =
        exact_reuse.cache_node_ids + query_id * exact_reuse.cache_capacity;
    float* distances =
        exact_reuse.cache_distances + query_id * exact_reuse.cache_capacity;
    const std::uint32_t mask = exact_reuse.cache_capacity - 1U;
    const std::uint32_t start = HashExactReuseNode(node_id) & mask;
    for (std::uint32_t probe = 0; probe < exact_reuse.cache_capacity; ++probe) {
        const std::uint32_t slot = (start + probe) & mask;
        const std::uint32_t previous =
            atomicCAS(node_ids + slot, kInvalidNodeId, node_id);
        if (previous == kInvalidNodeId || previous == node_id) {
            distances[slot] = distance;
            if (previous == kInvalidNodeId) {
                atomicAdd(&stats->exact_reuse_inserts, 1U);
            }
            return;
        }
    }
    atomicAdd(&stats->exact_reuse_overflows, 1U);
}

__device__ __forceinline__ void CopyTopologyNeighborsForWarp(
    const std::uint32_t* topology,
    const void* topology_ssd,
    std::uint64_t cached_node_count,
    std::uint32_t nodes_per_page,
    std::uint64_t num_nodes,
    std::uint32_t degree,
    std::uint32_t current_node,
    std::size_t query_id,
    const DeviceTraversalExactReuseConfig& exact_reuse,
    DeviceSearchStats* stats,
    DeviceProfileCycles* profile,
    std::uint32_t* shared_dst,
    std::uint32_t lane_id,
    std::uint32_t active_mask) {
    const std::size_t node_bytes =
        static_cast<std::size_t>(degree) * sizeof(std::uint32_t);
    const bool uncached =
        static_cast<std::uint64_t>(current_node) >= cached_node_count;
    if (uncached && exact_reuse.enabled != 0U) {
        const std::size_t page_id =
            static_cast<std::size_t>(current_node) / exact_reuse.nodes_per_page;
        const std::size_t slot_id =
            static_cast<std::size_t>(current_node) -
            page_id * static_cast<std::size_t>(exact_reuse.nodes_per_page);
        const std::size_t page_offset = page_id * kDefaultPageSizeBytes;
        const float* query =
            exact_reuse.queries + query_id * exact_reuse.query_dim;
        const unsigned long long read_begin = clock64();
        const float distance = ReadCombinedNodeFromBam(
            exact_reuse.combined_ssd, page_offset,
            slot_id * static_cast<std::size_t>(exact_reuse.node_bytes),
            exact_reuse.query_dim, exact_reuse.vector_bytes, num_nodes, degree,
            query, shared_dst, lane_id, active_mask);
        if (lane_id == 0) {
            atomicAdd(&profile->combined_node_read_cycles, clock64() - read_begin);
            const unsigned long long insert_begin = clock64();
            InsertExactReuse(exact_reuse, query_id, current_node, distance, stats);
            atomicAdd(&profile->exact_reuse_insert_cycles,
                      clock64() - insert_begin);
        }
        return;
    }
    if (topology_ssd != nullptr && uncached) {
        const std::size_t page_id =
            static_cast<std::size_t>(current_node) / nodes_per_page;
        const std::size_t slot_id =
            static_cast<std::size_t>(current_node) -
            page_id * static_cast<std::size_t>(nodes_per_page);
        const std::size_t page_offset = page_id * kDefaultPageSizeBytes;
        CopyTopologyNeighborsFromBam(topology_ssd, page_offset, slot_id * node_bytes,
                                     degree, shared_dst, lane_id, active_mask);
        return;
    }

    const void* topology_src =
        topology + static_cast<std::size_t>(current_node) * degree;
    copyFromHostCacheVec(topology_src, shared_dst, node_bytes, active_mask);
}

__device__ __forceinline__ std::uint32_t HashVisited
(int hash_id, std::uint32_t value) {
    value ^= value >> 16;
    value *= kVisitedHashSeeds[hash_id << 1];
    value ^= value >> 13;
    value *= kVisitedHashSeeds[(hash_id << 1) + 1];
    value ^= value >> 16;
    return value % static_cast<std::uint32_t>(kVisitedHashWords << 5);
}

__device__ __forceinline__ void MarkGroundTruthVisited(
    std::uint32_t node_id,
    const std::uint32_t* query_gt_ids,
    std::uint32_t gt_topk,
    std::uint32_t* query_gt_hit_mask) {
    if (query_gt_ids == nullptr || query_gt_hit_mask == nullptr || gt_topk == 0U ||
        gt_topk > 32U) {
        return;
    }
    for (std::uint32_t i = 0; i < gt_topk; ++i) {
        if (query_gt_ids[i] == node_id) {
            atomicOr(query_gt_hit_mask, 1U << i);
        }
    }
}

__device__ __forceinline__ void VisitedAdd(std::uint32_t value, std::uint32_t* visited_hash) {
    for (std::size_t i = 0; i < kVisitedHashFunctions; ++i) {
        const std::uint32_t bit = HashVisited(static_cast<int>(i), value);
        const std::uint32_t mask = 1U << (bit / kVisitedHashWords);
        atomicOr(&visited_hash[bit % kVisitedHashWords], mask);
    }
}

__device__ __forceinline__ bool VisitedTest(std::uint32_t value,
                                            const std::uint32_t* visited_hash) {
    bool ok = true;
    for (std::size_t i = 0; i < kVisitedHashFunctions; ++i) {
        const std::uint32_t bit = HashVisited(static_cast<int>(i), value);
        ok &= ((visited_hash[bit % kVisitedHashWords] >> (bit / kVisitedHashWords)) & 1U) != 0U;
    }
    return ok;
}

template <std::size_t kFixedNumChunks = kNoFixedNumChunks>
__device__ __forceinline__ float PqDistanceForVertex(std::uint32_t node_id,
                                                     std::size_t num_chunks,
                                                     const float* query_tables,
                                                     const std::uint8_t* pq_codes,
                                                     std::size_t lane_id) {
    float distance = 0.0f;
    if constexpr (kFixedNumChunks == kSpecializedSearchPqNumChunks) {
        const std::uint8_t* node_codes =
            pq_codes + static_cast<std::size_t>(node_id) * kSpecializedSearchPqNumChunks;
        if (lane_id < kWarpSize) {
            distance += query_tables[lane_id * kNumPqCentroids + node_codes[lane_id]];
        }
        const float partial_sum = WarpReduceSum(distance);
        float tail_distance = 0.0f;
        if (lane_id + kWarpSize < kSpecializedSearchPqNumChunks) {
            const std::size_t chunk = lane_id + kWarpSize;
            tail_distance += query_tables[chunk * kNumPqCentroids + node_codes[chunk]];
        }
        const float tail_sum = WarpReduceSum(tail_distance);
        return partial_sum + tail_sum;
    } else {
        float partial_sum = 0.0f;
        const std::uint8_t* node_codes =
            pq_codes + static_cast<std::size_t>(node_id) * num_chunks;
        for (std::size_t chunk_base = 0; chunk_base < num_chunks; chunk_base += kWarpSize) {
            float chunk_group_distance = 0.0f;
            const std::size_t chunk = chunk_base + lane_id;
            if (chunk < num_chunks) {
                const std::uint8_t center_id = node_codes[chunk];
                chunk_group_distance += query_tables[chunk * kNumPqCentroids + center_id];
            }
            chunk_group_distance = WarpReduceSum(chunk_group_distance);
            partial_sum += __shfl_sync(kFullMask, chunk_group_distance, 0);
        }
        return partial_sum;
    }
}

__device__ __forceinline__ float PqDistanceForVertexBang(std::uint32_t node_id,
                                                         std::size_t num_chunks,
                                                         const float* query_tables,
                                                         const std::uint8_t* pq_codes,
                                                         std::size_t lane_id) {
    const std::size_t lane_in_group = lane_id & 7U;
    const std::uint8_t* node_codes =
        pq_codes + static_cast<std::size_t>(node_id) * num_chunks;
    float distance = 0.0f;
    for (std::size_t chunk = lane_in_group; chunk < num_chunks; chunk += 8U) {
        distance += query_tables[chunk * kNumPqCentroids + node_codes[chunk]];
    }
    return GroupReduceSumWidth8(distance);
}

__device__ __forceinline__ float PqDistanceForVertexGustann(std::uint32_t node_id,
                                                            std::size_t num_chunks,
                                                            const float* query_tables,
                                                            const std::uint8_t* pq_codes) {
    const std::uint8_t* node_codes =
        pq_codes + static_cast<std::size_t>(node_id) * num_chunks;
    float distance = 0.0f;
    for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
        distance += query_tables[chunk * kNumPqCentroids + node_codes[chunk]];
    }
    return distance;
}

__device__ __forceinline__ float PqDistanceForVertexGustannTiled(
    std::uint32_t node_id,
    std::size_t num_chunks,
    const std::uint8_t* pq_codes,
    const float* staged_query_tables,
    std::size_t chunk_base,
    std::size_t staged_chunk_count) {
    const std::uint8_t* node_codes =
        pq_codes + static_cast<std::size_t>(node_id) * num_chunks + chunk_base;
    float distance = 0.0f;
    for (std::size_t local_chunk = 0; local_chunk < staged_chunk_count; ++local_chunk) {
        distance += staged_query_tables[local_chunk * kNumPqCentroids + node_codes[local_chunk]];
    }
    return distance;
}

__device__ __forceinline__ void PrefetchQueryTableTileAsync(float* shared_query_tables,
                                                            const float* query_tables,
                                                            std::size_t value_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    const auto* query_tables_vec = reinterpret_cast<const float4*>(query_tables);
    auto* shared_query_tables_vec = reinterpret_cast<float4*>(shared_query_tables);
    const std::size_t value_count_vec = value_count / 4U;
    for (std::size_t vec_index = threadIdx.x; vec_index < value_count_vec;
         vec_index += blockDim.x) {
        const unsigned shared_address = static_cast<unsigned>(
            __cvta_generic_to_shared(shared_query_tables_vec + vec_index));
        asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(shared_address), "l"(query_tables_vec + vec_index));
    }
    asm volatile("cp.async.commit_group;\n" ::);
#else
    for (std::size_t value_index = threadIdx.x; value_index < value_count;
         value_index += blockDim.x) {
        shared_query_tables[value_index] = query_tables[value_index];
    }
#endif
}

__device__ __forceinline__ void WaitForPrefetchedQueryTableTile() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
    __syncthreads();
}

__device__ void BitonicSortCandidates(DeviceCandidate* candidates, std::size_t count) {
    for (std::size_t k = 2; k <= count; k <<= 1) {
        for (std::size_t j = k >> 1; j > 0; j >>= 1) {
            for (std::size_t idx = threadIdx.x; idx < count; idx += blockDim.x) {
                const std::size_t other = idx ^ j;
                if (other <= idx || other >= count) {
                    continue;
                }
                const bool ascending = (idx & k) == 0;
                const bool should_swap =
                    ascending ? CandidateLess(candidates[other], candidates[idx])
                              : CandidateLess(candidates[idx], candidates[other]);
                if (should_swap) {
                    const DeviceCandidate temporary = candidates[idx];
                    candidates[idx] = candidates[other];
                    candidates[other] = temporary;
                }
            }
            __syncthreads();
        }
    }
}

__device__ std::uint32_t CountValidCandidates(const DeviceCandidate* candidates,
                                              std::size_t count) {
    std::uint32_t valid = 0;
    for (std::size_t i = 0; i < count; ++i) {
        valid += IsValid(candidates[i]) ? 1U : 0U;
    }
    return valid;
}

__device__ void ClearCandidates(DeviceCandidate* candidates, std::size_t count) {
    for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
        candidates[i] = InvalidDeviceCandidate();
    }
    __syncthreads();
}

__device__ void ClearVisitedHashWords(std::uint32_t* visited_hash) {
    for (std::size_t i = threadIdx.x; i < kVisitedHashWords; i += blockDim.x) {
        visited_hash[i] = 0U;
    }
    __syncthreads();
}

__device__ void ClearBitWords(std::uint32_t* bits, std::size_t count) {
    for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
        bits[i] = 0U;
    }
    __syncthreads();
}

__device__ void RebuildPrefixState(const DeviceCandidate* candidates,
                                   std::size_t prefix_count,
                                   std::uint32_t* unexpanded_bits,
                                   std::size_t unexpanded_words,
                                   bool rebuild_hash,
                                   std::uint32_t* visited_hash) {
    ClearBitWords(unexpanded_bits, unexpanded_words);
    if (rebuild_hash) {
        ClearVisitedHashWords(visited_hash);
    }
    for (std::size_t i = threadIdx.x; i < prefix_count; i += blockDim.x) {
        const DeviceCandidate candidate = candidates[i];
        if (!IsValid(candidate)) {
            continue;
        }
        if (!candidate.expanded()) {
            atomicOr(&unexpanded_bits[i / 32U], 1U << (i % 32U));
        }
        if (rebuild_hash) {
            VisitedAdd(candidate.raw_node_id(), visited_hash);
        }
    }
    __syncthreads();
}

__device__ bool IsAnyBitSet(const std::uint32_t* bits, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        if (bits[i] != 0U) {
            return true;
        }
    }
    return false;
}

__device__ std::uint32_t CountBitsInRange(const std::uint32_t* bits,
                                          std::uint32_t begin_bit,
                                          std::uint32_t end_bit) {
    if (begin_bit >= end_bit) {
        return 0U;
    }
    std::uint32_t count = 0U;
    const std::uint32_t begin_word = begin_bit / 32U;
    const std::uint32_t end_word = (end_bit - 1U) / 32U;
    for (std::uint32_t word_index = begin_word; word_index <= end_word; ++word_index) {
        std::uint32_t word = bits[word_index];
        const std::uint32_t word_begin = word_index == begin_word ? (begin_bit % 32U) : 0U;
        const std::uint32_t word_end =
            word_index == end_word ? ((end_bit - 1U) % 32U) + 1U : 32U;
        const std::uint32_t left_mask =
            word_begin == 0U ? 0xffffffffU : (~0U << word_begin);
        const std::uint32_t right_mask =
            word_end == 32U ? 0xffffffffU : ((1U << word_end) - 1U);
        word &= (left_mask & right_mask);
        count += static_cast<std::uint32_t>(__popc(word));
    }
    return count;
}

__device__ std::uint32_t FindFirstSetBit(const std::uint32_t* bits, std::uint32_t bit_limit) {
    const std::uint32_t word_count = (bit_limit + 31U) / 32U;
    for (std::uint32_t word_index = 0; word_index < word_count; ++word_index) {
        std::uint32_t word = bits[word_index];
        if (word_index + 1U == word_count && (bit_limit % 32U) != 0U) {
            word &= ((1U << (bit_limit % 32U)) - 1U);
        }
        if (word == 0U) {
            continue;
        }
        return word_index * 32U + static_cast<std::uint32_t>(__ffs(static_cast<int>(word)) - 1);
    }
    return bit_limit;
}

__device__ float ComputeTopKSetChurn(const DeviceCandidate* candidates,
                                     const std::uint32_t* previous_topk_ids,
                                     std::uint32_t top_k,
                                     bool has_previous) {
    if (!has_previous) {
        return 1.0f;
    }
    std::uint32_t changed = 0U;
    for (std::uint32_t i = 0; i < top_k; ++i) {
        const std::uint32_t current_id = candidates[i].raw_node_id();
        bool found = false;
        for (std::uint32_t j = 0; j < top_k; ++j) {
            if (previous_topk_ids[j] == current_id) {
                found = true;
                break;
            }
        }
        changed += found ? 0U : 1U;
    }
    return static_cast<float>(changed) / static_cast<float>(top_k);
}

__device__ float EvaluateLearnedStopLogit(const DeviceLearnedStopConfig& config,
                                          const float* features) {
    float logit = config.bias;
    logit = fmaf(features[0], config.linear_weights[0], logit);
    logit = fmaf(features[1], config.linear_weights[1], logit);
    logit = fmaf(features[2], config.linear_weights[2], logit);
    logit = fmaf(features[3], config.linear_weights[3], logit);
    logit = fmaf(features[4], config.linear_weights[4], logit);
    logit = fmaf(features[5], config.linear_weights[5], logit);
    logit = fmaf(features[6], config.linear_weights[6], logit);
    logit = fmaf(features[7], config.linear_weights[7], logit);
    logit = fmaf(features[8], config.linear_weights[8], logit);
    logit = fmaf(features[9], config.linear_weights[9], logit);
    logit = fmaf(features[10], config.linear_weights[10], logit);
    return fmaf(features[11], config.linear_weights[11], logit);
}

__device__ __forceinline__ float EvaluateLearnedStopLogitFastCore7(
    const DeviceLearnedStopConfig& config,
    float visited_frac,
    float best_unexpanded_rank_frac,
    float dk,
    float gap_prefix,
    float gap32,
    float delta_boundary,
    float topk_changed_flag) {
    float logit = config.bias;
    logit = fmaf(visited_frac, config.linear_weights[2], logit);
    logit = fmaf(best_unexpanded_rank_frac, config.linear_weights[3], logit);
    logit = fmaf(dk, config.linear_weights[6], logit);
    logit = fmaf(gap_prefix, config.linear_weights[7], logit);
    logit = fmaf(gap32, config.linear_weights[8], logit);
    logit = fmaf(delta_boundary, config.linear_weights[10], logit);
    return fmaf(topk_changed_flag, config.linear_weights[11], logit);
}

__device__ void MaybeEvaluateLearnedStop(const DeviceLearnedStopConfig& learned_stop,
                                         const DeviceCandidate* candidates,
                                         const std::uint32_t* unexpanded_bits,
                                         std::uint32_t top_l,
                                         const DeviceSearchStats& stats,
                                         std::uint32_t* learned_stop_stage,
                                         std::uint32_t* learned_stop_action,
                                         std::uint32_t* learned_stop_has_prev,
                                         float* learned_stop_prev_dk,
                                         float* learned_stop_prev_boundary,
                                         std::uint32_t* learned_stop_prev_topk_ids,
                                         std::uint32_t* learned_stop_topk_changed,
                                         DeviceProfileCycles* profile) {
    if (threadIdx.x != 0 || learned_stop.enabled == 0U) {
        return;
    }

    const unsigned long long model_begin = clock64();
    *learned_stop_action = 0U;
    while (*learned_stop_stage < learned_stop.num_stages) {
        const std::uint32_t active_prefix =
            learned_stop.stage_prefixes[*learned_stop_stage];
        if (active_prefix == 0U || active_prefix >= top_l ||
            active_prefix < learned_stop.top_k) {
            ++(*learned_stop_stage);
            continue;
        }
        if (stats.valid_candidates < top_l ||
            stats.valid_candidates < active_prefix ||
            stats.valid_candidates < learned_stop.top_k ||
            stats.expanded_nodes < active_prefix) {
            break;
        }

        const float dk = candidates[learned_stop.top_k - 1U].distance;
        const float boundary = candidates[active_prefix - 1U].distance;
        const std::uint32_t next64_end = std::min(active_prefix + 64U, top_l);
        const std::uint32_t next256_end = std::min(active_prefix + 256U, top_l);
        const std::uint32_t next64_span =
            next64_end > active_prefix ? (next64_end - active_prefix) : 1U;
        const std::uint32_t next256_span =
            next256_end > active_prefix ? (next256_end - active_prefix) : 1U;
        const std::uint32_t gap32_index =
            std::min(std::max(learned_stop.top_k - 1U, 31U), active_prefix - 1U);
        const bool has_previous = *learned_stop_has_prev != 0U;

        if (learned_stop.feature_variant == kTraversalLearnedStopFeatureVariantCore7) {
            const unsigned long long feature_begin = clock64();
            const unsigned long long find_first_set_begin = clock64();
            const std::uint32_t best_unexpanded_rank =
                FindFirstSetBit(unexpanded_bits, top_l);
            profile->learned_stop_find_first_set_cycles += clock64() - find_first_set_begin;
            const float visited_frac =
                static_cast<float>(stats.visited_nodes) * learned_stop.inv_top_l;
            const float best_unexpanded_rank_frac =
                static_cast<float>(best_unexpanded_rank) * learned_stop.inv_top_l;
            const float gap_prefix = boundary - dk;
            const float gap32 = candidates[gap32_index].distance - dk;
            const float delta_boundary =
                has_previous ? (*learned_stop_prev_boundary - boundary) : 0.0f;
            const unsigned long long topk_churn_begin = clock64();
            const float topk_changed_flag =
                *learned_stop_topk_changed != 0U ? 1.0f : 0.0f;
            profile->learned_stop_topk_churn_cycles += clock64() - topk_churn_begin;
            profile->learned_stop_feature_cycles += clock64() - feature_begin;

            const unsigned long long logit_begin = clock64();
            const float logit = EvaluateLearnedStopLogitFastCore7(
                learned_stop, visited_frac, best_unexpanded_rank_frac, dk,
                gap_prefix, gap32, delta_boundary, topk_changed_flag);
            profile->learned_stop_logit_eval_cycles += clock64() - logit_begin;
            *learned_stop_prev_dk = dk;
            *learned_stop_prev_boundary = boundary;
            *learned_stop_has_prev = 1U;
            *learned_stop_topk_changed = 0U;
            if (logit < learned_stop.threshold_logit) {
                if (learned_stop.shadow_mode != 0U) {
                    *learned_stop_stage = learned_stop.num_stages;
                    break;
                }
                *learned_stop_action = 1U;
                break;
            }
            ++(*learned_stop_stage);
            continue;
        }

        float features[kLearnedStopFeatureCount];
        const unsigned long long feature_begin = clock64();
        features[0] = learned_stop.feature_mask[0] != 0U
                          ? static_cast<float>(active_prefix) * learned_stop.inv_top_l
                          : 0.0f;
        features[1] = learned_stop.feature_mask[1] != 0U
                          ? static_cast<float>(stats.expanded_nodes) * learned_stop.inv_top_l
                          : 0.0f;
        features[2] = learned_stop.feature_mask[2] != 0U
                          ? static_cast<float>(stats.visited_nodes) * learned_stop.inv_top_l
                          : 0.0f;
        const bool need_best_unexpanded_rank =
            learned_stop.feature_mask[3] != 0U ||
            (learned_stop.feature_mask[4] != 0U &&
             learned_stop.next64_mode == kNext64ModeFlag);
        std::uint32_t best_unexpanded_rank = top_l;
        if (need_best_unexpanded_rank) {
            const unsigned long long find_first_set_begin = clock64();
            best_unexpanded_rank = FindFirstSetBit(unexpanded_bits, top_l);
            features[3] = learned_stop.feature_mask[3] != 0U
                              ? static_cast<float>(best_unexpanded_rank) * learned_stop.inv_top_l
                              : 0.0f;
            profile->learned_stop_find_first_set_cycles += clock64() - find_first_set_begin;
        } else {
            features[3] = 0.0f;
        }
        if (learned_stop.feature_mask[4] != 0U || learned_stop.feature_mask[5] != 0U) {
            const unsigned long long count_bits_begin = clock64();
            if (learned_stop.feature_mask[4] != 0U) {
                if (learned_stop.next64_mode == kNext64ModeFlag) {
                    features[4] = best_unexpanded_rank < next64_end ? 1.0f : 0.0f;
                } else {
                    features[4] =
                        static_cast<float>(CountBitsInRange(unexpanded_bits, active_prefix, next64_end)) /
                        static_cast<float>(next64_span);
                }
            } else {
                features[4] = 0.0f;
            }
            if (learned_stop.feature_mask[5] != 0U) {
                features[5] =
                    static_cast<float>(CountBitsInRange(unexpanded_bits, active_prefix, next256_end)) /
                    static_cast<float>(next256_span);
            } else {
                features[5] = 0.0f;
            }
            profile->learned_stop_count_bits_cycles += clock64() - count_bits_begin;
        } else {
            features[4] = 0.0f;
            features[5] = 0.0f;
        }
        features[6] = learned_stop.feature_mask[6] != 0U ? dk : 0.0f;
        features[7] = learned_stop.feature_mask[7] != 0U ? (boundary - dk) : 0.0f;
        features[8] =
            learned_stop.feature_mask[8] != 0U ? (candidates[gap32_index].distance - dk) : 0.0f;
        features[9] =
            learned_stop.feature_mask[9] != 0U && has_previous ? (*learned_stop_prev_dk - dk)
                                                                : 0.0f;
        features[10] = learned_stop.feature_mask[10] != 0U && has_previous
                           ? (*learned_stop_prev_boundary - boundary)
                           : 0.0f;
        const unsigned long long topk_churn_begin = clock64();
        if (learned_stop.feature_mask[11] == 0U) {
            features[11] = 0.0f;
        } else if (learned_stop.topk_churn_mode == kTopkChurnModeFlag) {
            features[11] = *learned_stop_topk_changed != 0U ? 1.0f : 0.0f;
        } else {
            features[11] = ComputeTopKSetChurn(candidates, learned_stop_prev_topk_ids,
                                               learned_stop.top_k, has_previous);
        }
        profile->learned_stop_topk_churn_cycles += clock64() - topk_churn_begin;
        profile->learned_stop_feature_cycles += clock64() - feature_begin;

        const unsigned long long logit_begin = clock64();
        const float logit = EvaluateLearnedStopLogit(learned_stop, features);
        profile->learned_stop_logit_eval_cycles += clock64() - logit_begin;
        *learned_stop_prev_dk = dk;
        *learned_stop_prev_boundary = boundary;
        *learned_stop_has_prev = 1U;
        *learned_stop_topk_changed = 0U;
        if (learned_stop.feature_mask[11] != 0U &&
            learned_stop.topk_churn_mode == kTopkChurnModeRatio) {
            for (std::uint32_t i = 0; i < learned_stop.top_k; ++i) {
                learned_stop_prev_topk_ids[i] = candidates[i].raw_node_id();
            }
        }
        if (logit < learned_stop.threshold_logit) {
            if (learned_stop.shadow_mode != 0U) {
                *learned_stop_stage = learned_stop.num_stages;
                break;
            }
            *learned_stop_action = 1U;
            break;
        }
        ++(*learned_stop_stage);
    }
    profile->learned_stop_model_cycles += clock64() - model_begin;
}

template <std::size_t kFixedNumChunks = kNoFixedNumChunks>
__device__ void ComputeFrontierDistances(DeviceCandidate* frontier,
                                         std::size_t frontier_count,
                                         std::size_t num_chunks,
                                         const float* query_tables,
                                         const std::uint8_t* pq_codes,
                                         float* shared_query_tables,
                                         std::size_t lane_id,
                                         std::size_t warp_id,
                                         std::uint32_t frontier_pq_warps,
                                         std::uint32_t frontier_pq_mode,
                                         std::uint32_t gustann_tile_chunks,
                                         DeviceProfileCycles* profile) {
    if (frontier_pq_mode == kFrontierPqModeBang) {
        unsigned long long compute_begin = 0;
        if (threadIdx.x == 0) {
            compute_begin = clock64();
        }
        const std::uint32_t group_id = static_cast<std::uint32_t>(threadIdx.x / 8U);
        const std::uint32_t lane_in_group = static_cast<std::uint32_t>(threadIdx.x & 7U);
        const std::uint32_t group_count = static_cast<std::uint32_t>(blockDim.x / 8U);
        for (std::size_t base = 0; base < frontier_count; base += group_count) {
            const std::size_t frontier_index = base + group_id;
            if (frontier_index >= frontier_count) {
                continue;
            }
            const std::uint32_t node_id = frontier[frontier_index].node_id;
            const float distance = !PackedNodeIdValid(node_id)
                                       ? 0.0f
                                       : PqDistanceForVertexBang(RawNodeId(node_id), num_chunks,
                                                                 query_tables, pq_codes,
                                                                 lane_in_group);
            if (lane_in_group == 0U) {
                frontier[frontier_index].distance = !PackedNodeIdValid(node_id)
                                                        ? std::numeric_limits<float>::infinity()
                                                        : distance;
            }
        }
        if (threadIdx.x == 0) {
            profile->pq_compute_cycles += clock64() - compute_begin;
        }
        return;
    }
    if (frontier_pq_mode == kFrontierPqModeGustann) {
        if (gustann_tile_chunks >= 2U && gustann_tile_chunks <= kMaxFrontierPqTileChunks) {
            const std::size_t tile_chunks = static_cast<std::size_t>(gustann_tile_chunks);
            const std::size_t tile_values = tile_chunks * kNumPqCentroids;
            float* staged_query_tables[2] = {
                shared_query_tables,
                shared_query_tables + tile_values,
            };
            for (std::size_t frontier_base = 0; frontier_base < frontier_count;
                 frontier_base += blockDim.x) {
                const std::size_t frontier_index = frontier_base + threadIdx.x;
                const bool active = frontier_index < frontier_count;
                const std::uint32_t node_id = active ? frontier[frontier_index].node_id
                                                     : kInvalidNodeId;
                float distance = 0.0f;
                std::size_t chunk_base = 0;
                std::size_t staged_chunk_count = std::min(tile_chunks, num_chunks);
                unsigned long long stage_begin = 0;
                if (threadIdx.x == 0) {
                    stage_begin = clock64();
                }
                PrefetchQueryTableTileAsync(staged_query_tables[0], query_tables,
                                            staged_chunk_count * kNumPqCentroids);
                if (threadIdx.x == 0) {
                    profile->pq_prefetch_issue_cycles += clock64() - stage_begin;
                    stage_begin = clock64();
                }
                WaitForPrefetchedQueryTableTile();
                if (threadIdx.x == 0) {
                    profile->pq_prefetch_wait_cycles += clock64() - stage_begin;
                }
                std::uint32_t current_stage = 0U;
                while (chunk_base < num_chunks) {
                    const std::size_t next_chunk_base = chunk_base + staged_chunk_count;
                    const std::size_t next_staged_chunk_count =
                        next_chunk_base < num_chunks
                            ? std::min(tile_chunks, num_chunks - next_chunk_base)
                            : 0U;
                    if (next_staged_chunk_count != 0U) {
                        if (threadIdx.x == 0) {
                            stage_begin = clock64();
                        }
                        PrefetchQueryTableTileAsync(
                            staged_query_tables[current_stage ^ 1U],
                            query_tables + next_chunk_base * kNumPqCentroids,
                            next_staged_chunk_count * kNumPqCentroids);
                        if (threadIdx.x == 0) {
                            profile->pq_prefetch_issue_cycles += clock64() - stage_begin;
                        }
                    }
                    if (threadIdx.x == 0) {
                        stage_begin = clock64();
                    }
                    if (active && PackedNodeIdValid(node_id)) {
                        distance += PqDistanceForVertexGustannTiled(
                            RawNodeId(node_id), num_chunks, pq_codes,
                            staged_query_tables[current_stage],
                            chunk_base,
                            staged_chunk_count);
                    }
                    if (threadIdx.x == 0) {
                        profile->pq_compute_cycles += clock64() - stage_begin;
                    }
                    if (next_staged_chunk_count != 0U) {
                        if (threadIdx.x == 0) {
                            stage_begin = clock64();
                        }
                        WaitForPrefetchedQueryTableTile();
                        if (threadIdx.x == 0) {
                            profile->pq_prefetch_wait_cycles += clock64() - stage_begin;
                        }
                        current_stage ^= 1U;
                    }
                    chunk_base = next_chunk_base;
                    staged_chunk_count = next_staged_chunk_count;
                }
                if (active) {
                    frontier[frontier_index].distance = !PackedNodeIdValid(node_id)
                                                            ? std::numeric_limits<float>::infinity()
                                                            : distance;
                }
            }
            return;
        }
        unsigned long long compute_begin = 0;
        if (threadIdx.x == 0) {
            compute_begin = clock64();
        }
        for (std::size_t frontier_index = threadIdx.x; frontier_index < frontier_count;
             frontier_index += blockDim.x) {
            const std::uint32_t node_id = frontier[frontier_index].node_id;
            frontier[frontier_index].distance = !PackedNodeIdValid(node_id)
                                                    ? std::numeric_limits<float>::infinity()
                                                    : PqDistanceForVertexGustann(
                                                          RawNodeId(node_id), num_chunks,
                                                          query_tables, pq_codes);
        }
        if (threadIdx.x == 0) {
            profile->pq_compute_cycles += clock64() - compute_begin;
        }
        return;
    }
    const std::uint32_t available_warps =
        static_cast<std::uint32_t>(blockDim.x / kWarpSize);
    const std::uint32_t active_warps =
        std::max(1U, std::min(frontier_pq_warps, available_warps));
    if (warp_id >= active_warps) {
        return;
    }
    unsigned long long compute_begin = 0;
    if (threadIdx.x == 0) {
        compute_begin = clock64();
    }
    if (active_warps == 1U) {
        for (std::size_t i = 0; i < frontier_count; ++i) {
            const std::uint32_t node_id = frontier[i].node_id;
            const float distance = !PackedNodeIdValid(node_id)
                                       ? 0.0f
                                       : PqDistanceForVertex<kFixedNumChunks>(
                                             RawNodeId(node_id), num_chunks, query_tables, pq_codes,
                                             lane_id);
            if (lane_id == 0) {
                frontier[i].distance = !PackedNodeIdValid(node_id)
                                           ? std::numeric_limits<float>::infinity()
                                           : distance;
            }
            __syncwarp();
        }
        if (threadIdx.x == 0) {
            profile->pq_compute_cycles += clock64() - compute_begin;
        }
        return;
    }
    for (std::size_t base = 0; base < frontier_count; base += active_warps) {
        const std::size_t frontier_index = base + warp_id;
        if (frontier_index >= frontier_count) {
            continue;
        }
        const std::uint32_t node_id = frontier[frontier_index].node_id;
        const float distance = !PackedNodeIdValid(node_id)
                                   ? 0.0f
                                   : PqDistanceForVertex<kFixedNumChunks>(RawNodeId(node_id),
                                                                          num_chunks,
                                                                          query_tables, pq_codes,
                                                                          lane_id);
        if (lane_id == 0) {
            frontier[frontier_index].distance = !PackedNodeIdValid(node_id)
                                                    ? std::numeric_limits<float>::infinity()
                                                    : distance;
        }
    }
    if (threadIdx.x == 0) {
        profile->pq_compute_cycles += clock64() - compute_begin;
    }
}

__device__ std::uint32_t MergeFrontierIntoCandidates(DeviceCandidate* candidates,
                                                     std::size_t candidate_capacity,
                                                     std::uint32_t valid_count,
                                                     DeviceCandidate* frontier,
                                                     std::size_t frontier_capacity,
                                                     DeviceCandidate* merged_candidates,
                                                     std::uint32_t top_k,
                                                     std::uint32_t* out_topk_changed,
                                                     std::uint32_t* out_accepted_frontier,
                                                     DeviceProfileCycles* profile) {
    unsigned long long stage_begin = 0;
    if (threadIdx.x == 0) {
        stage_begin = clock64();
    }
    __syncthreads();
    BitonicSortCandidates(frontier, frontier_capacity);
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned long long stage_end = clock64();
        const unsigned long long delta = stage_end - stage_begin;
        profile->frontier_sort_cycles += delta;
        profile->queue_cycles += delta;
        stage_begin = stage_end;
    }
    __syncthreads();
    const std::size_t frontier_valid = CountValidCandidates(frontier, frontier_capacity);
    const std::size_t new_valid_count =
        std::min(candidate_capacity, static_cast<std::size_t>(valid_count) + frontier_valid);
    std::size_t accepted = 0;
    while (accepted < frontier_valid && accepted < new_valid_count) {
        const std::size_t candidate_index = new_valid_count - 1U - accepted;
        if (!CandidateLess(frontier[accepted], candidates[candidate_index])) {
            break;
        }
        ++accepted;
    }
    if (threadIdx.x == 0 && out_accepted_frontier != nullptr) {
        *out_accepted_frontier = static_cast<std::uint32_t>(accepted);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned long long stage_end = clock64();
        const unsigned long long delta = stage_end - stage_begin;
        profile->tail_merge_cycles += delta;
        profile->queue_cycles += delta;
        stage_begin = stage_end;
    }
    __syncthreads();
    if (threadIdx.x == 0 && accepted != 0U) {
        const std::size_t kept_candidate_count = new_valid_count - accepted;
        std::ptrdiff_t candidate_index = static_cast<std::ptrdiff_t>(kept_candidate_count) - 1;
        std::ptrdiff_t frontier_index = static_cast<std::ptrdiff_t>(accepted) - 1;
        std::ptrdiff_t output_index = static_cast<std::ptrdiff_t>(new_valid_count) - 1;
        bool topk_changed = false;
        while (frontier_index >= 0) {
            if (candidate_index >= 0 &&
                !CandidateLess(candidates[static_cast<std::size_t>(candidate_index)],
                               frontier[static_cast<std::size_t>(frontier_index)])) {
                candidates[static_cast<std::size_t>(output_index)] =
                    candidates[static_cast<std::size_t>(candidate_index)];
                --candidate_index;
            } else {
                candidates[static_cast<std::size_t>(output_index)] =
                    frontier[static_cast<std::size_t>(frontier_index)];
                if (output_index >= 0 &&
                    static_cast<std::uint32_t>(output_index) < top_k) {
                    topk_changed = true;
                }
                --frontier_index;
            }
            --output_index;
        }
        if (out_topk_changed != nullptr) {
            *out_topk_changed = topk_changed ? 1U : 0U;
        }
    } else if (threadIdx.x == 0 && out_topk_changed != nullptr) {
        *out_topk_changed = 0U;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned long long stage_end = clock64();
        const unsigned long long delta = stage_end - stage_begin;
        profile->candidate_sort_cycles += delta;
        profile->queue_cycles += delta;
    }
    return static_cast<std::uint32_t>(new_valid_count);
}

__device__ std::uint32_t MergeFrontierIntoCandidatesHighL(DeviceCandidate* candidates,
                                                          std::size_t candidate_capacity,
                                                          std::uint32_t valid_count,
                                                          DeviceCandidate* frontier,
                                                          std::size_t frontier_capacity,
                                                          DeviceCandidate* merged_candidates,
                                                          std::uint32_t* frontier_output_positions,
                                                          std::uint32_t* frontier_output_count,
                                                          std::uint32_t top_k,
                                                          std::uint32_t* out_topk_changed,
                                                          std::uint32_t* out_accepted_frontier,
                                                          DeviceProfileCycles* profile) {
    unsigned long long stage_begin = 0;
    if (threadIdx.x == 0) {
        stage_begin = clock64();
    }
    __syncthreads();
    BitonicSortCandidates(frontier, frontier_capacity);
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned long long stage_end = clock64();
        const unsigned long long delta = stage_end - stage_begin;
        profile->frontier_sort_cycles += delta;
        profile->queue_cycles += delta;
        stage_begin = stage_end;
    }
    __syncthreads();
    const std::uint32_t frontier_valid =
        static_cast<std::uint32_t>(CountValidCandidates(frontier, frontier_capacity));
    const std::uint32_t new_valid_count = static_cast<std::uint32_t>(
        std::min(candidate_capacity,
                 static_cast<std::size_t>(valid_count) + frontier_valid));
    if (threadIdx.x == 0) {
        std::uint32_t accepted_frontier = 0U;
        bool topk_changed = false;
        for (std::uint32_t i = 0; i < frontier_valid; ++i) {
            const std::uint32_t output_position =
                UpperBoundCandidate(candidates, valid_count, frontier[i]) + i;
            if (output_position >= new_valid_count) {
                break;
            }
            frontier_output_positions[accepted_frontier++] = output_position;
            if (output_position < top_k) {
                topk_changed = true;
            }
        }
        *frontier_output_count = accepted_frontier;
        if (out_topk_changed != nullptr) {
            *out_topk_changed = topk_changed ? 1U : 0U;
        }
        if (out_accepted_frontier != nullptr) {
            *out_accepted_frontier = accepted_frontier;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned long long stage_end = clock64();
        const unsigned long long delta = stage_end - stage_begin;
        profile->tail_merge_cycles += delta;
        profile->queue_cycles += delta;
        stage_begin = stage_end;
    }
    __syncthreads();
    const std::uint32_t accepted_frontier = *frontier_output_count;
    for (std::size_t output_index = threadIdx.x; output_index < new_valid_count;
         output_index += blockDim.x) {
        std::uint32_t begin = 0U;
        std::uint32_t end = accepted_frontier;
        while (begin < end) {
            const std::uint32_t mid = (begin + end) >> 1U;
            if (frontier_output_positions[mid] <= output_index) {
                begin = mid + 1U;
            } else {
                end = mid;
            }
        }
        const std::uint32_t frontier_before = begin;
        if (frontier_before != 0U &&
            frontier_output_positions[frontier_before - 1U] == output_index) {
            merged_candidates[output_index] = frontier[frontier_before - 1U];
        } else {
            merged_candidates[output_index] = candidates[output_index - frontier_before];
        }
    }
    __syncthreads();
    for (std::size_t i = threadIdx.x; i < new_valid_count; i += blockDim.x) {
        candidates[i] = merged_candidates[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned long long stage_end = clock64();
        const unsigned long long delta = stage_end - stage_begin;
        profile->candidate_sort_cycles += delta;
        profile->queue_cycles += delta;
    }
    return new_valid_count;
}

__device__ void CaptureDebugSnapshot(
    std::size_t query_block_id,
    const DeviceCandidate* candidates,
    std::uint32_t capture_prefix,
    std::uint32_t merge_ordinal,
    std::uint32_t phase,
    std::uint32_t search_iteration,
    std::uint32_t frontier_valid,
    std::uint32_t accepted_frontier,
    std::uint32_t selected_count,
    std::uint64_t frontier_checksum,
    std::uint64_t visited_hash_checksum,
    const DeviceSearchStats& stats,
    std::uint32_t* debug_slot_shared,
    std::uint32_t* debug_snapshot_counts,
    DeviceTopologyDebugSnapshot* debug_snapshots,
    DeviceCandidate* debug_candidate_snapshots,
    std::uint32_t max_debug_snapshots) {
    if (debug_snapshot_counts == nullptr || debug_snapshots == nullptr ||
        debug_candidate_snapshots == nullptr || max_debug_snapshots == 0U ||
        capture_prefix == 0U) {
        return;
    }
    if (threadIdx.x == 0) {
        const std::uint32_t slot = atomicAdd(debug_snapshot_counts + query_block_id, 1U);
        *debug_slot_shared = slot;
        if (slot < max_debug_snapshots) {
            DeviceTopologyDebugSnapshot snapshot;
            snapshot.merge_ordinal = merge_ordinal;
            snapshot.phase = phase;
            snapshot.search_iteration = search_iteration;
            snapshot.frontier_valid = frontier_valid;
            snapshot.accepted_frontier = accepted_frontier;
            snapshot.selected_count = selected_count;
            snapshot.valid_candidates = stats.valid_candidates;
            snapshot.visited_nodes = stats.visited_nodes;
            snapshot.expanded_nodes = stats.expanded_nodes;
            snapshot.frontier_checksum = frontier_checksum;
            snapshot.visited_hash_checksum = visited_hash_checksum;
            debug_snapshots[query_block_id * max_debug_snapshots + slot] = snapshot;
        }
    }
    __syncthreads();
    const std::uint32_t slot = *debug_slot_shared;
    if (slot >= max_debug_snapshots) {
        return;
    }
    DeviceCandidate* debug_candidates =
        debug_candidate_snapshots +
        (query_block_id * max_debug_snapshots + slot) * static_cast<std::size_t>(capture_prefix);
    for (std::size_t i = threadIdx.x; i < capture_prefix; i += blockDim.x) {
        debug_candidates[i] = candidates[i];
    }
    __syncthreads();
}


template <std::size_t kFixedNumChunks = kNoFixedNumChunks>
__global__ void topology_microbatch_init_kernel(
    std::uint64_t num_nodes,
    const std::uint8_t* pq_codes,
    std::size_t num_chunks,
    const float* query_tables_base,
    std::uint32_t query_stride,
    const std::uint32_t* entry_offsets,
    const std::uint32_t* entry_ids,
    std::size_t query_offset,
    std::size_t num_queries,
    std::uint32_t top_l,
    std::uint32_t candidate_capacity,
    std::uint32_t frontier_slots,
    std::uint32_t frontier_capacity,
    std::uint32_t frontier_pq_warps,
    DeviceLearnedStopConfig,
    std::uint32_t* visited_hash_base,
    DeviceCandidate* out_candidates_base,
    DeviceSearchStats* out_stats_base,
    DeviceProfileCycles* out_profile_base,
    DeviceMicrobatchQueryState* query_states) {
    const DeviceLearnedStopConfig& learned_stop = kDeviceTraversalLearnedStopConfig;
    const std::size_t local_query_id = blockIdx.x;
    if (local_query_id >= num_queries) {
        return;
    }
    const std::size_t global_query_id = query_offset + local_query_id;
    extern __shared__ unsigned char shared_memory[];
    auto* candidates = reinterpret_cast<DeviceCandidate*>(shared_memory);
    auto* frontier = candidates + candidate_capacity;
    auto* staged_neighbors = reinterpret_cast<std::uint32_t*>(
        AlignPointer<vectype>(reinterpret_cast<void*>(frontier + frontier_capacity)));
    const std::size_t unexpanded_words = (top_l + 31U) / 32U;
    auto* unexpanded_bits = AlignPointer<std::uint32_t>(
        staged_neighbors + frontier_slots);
    auto* frontier_query_tables_shared = reinterpret_cast<float*>(
        AlignPointer<float4>(unexpanded_bits + unexpanded_words));

    __shared__ DeviceSearchStats stats;
    __shared__ DeviceProfileCycles profile;
    __shared__ std::uint32_t hash_iteration;
    __shared__ std::uint32_t learned_stop_topk_changed;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    const float* query_tables =
        query_tables_base + global_query_id * query_stride;
    const std::uint32_t entry_begin = entry_offsets[global_query_id];
    const std::uint32_t entry_end = entry_offsets[global_query_id + 1U];
    const std::uint32_t num_entries = entry_end - entry_begin;
    const std::uint32_t* query_entry_ids = entry_ids + entry_begin;
    std::uint32_t* visited_hash =
        visited_hash_base + local_query_id * kVisitedHashWords;
    DeviceCandidate* out_candidates =
        out_candidates_base + global_query_id * candidate_capacity;

    ClearVisitedHashWords(visited_hash);
    if (threadIdx.x == 0) {
        stats = DeviceSearchStats{};
        stats.first_full_prefix_iteration = kInvalidFullPrefixIteration;
        profile = DeviceProfileCycles{};
        hash_iteration = 0U;
        learned_stop_topk_changed = 0U;
    }
    __syncthreads();
    ClearCandidates(candidates, candidate_capacity);
    ClearCandidates(frontier, frontier_capacity);
    ClearBitWords(unexpanded_bits, unexpanded_words);

    const std::uint32_t entry_batches =
        frontier_slots == 0U ? 0U :
        (num_entries + frontier_slots - 1U) / frontier_slots;
    for (std::uint32_t batch = 0; batch < entry_batches; ++batch) {
        ClearCandidates(frontier, frontier_capacity);
        for (std::size_t i = threadIdx.x; i < frontier_slots;
             i += blockDim.x) {
            const std::size_t entry_index =
                batch * frontier_slots + i;
            if (entry_index >= num_entries) {
                continue;
            }
            const std::uint32_t node_id =
                query_entry_ids[entry_index];
            if (node_id >= num_nodes ||
                VisitedTest(node_id, visited_hash)) {
                continue;
            }
            frontier[i].set_raw_node_id(node_id, false);
        }
        __syncthreads();
        for (std::size_t i = threadIdx.x; i < frontier_slots;
             i += blockDim.x) {
            if (!IsValid(frontier[i])) {
                continue;
            }
            VisitedAdd(frontier[i].raw_node_id(), visited_hash);
            atomicAdd(&stats.visited_nodes, 1U);
        }
        __syncthreads();
        unsigned long long stage_begin = 0;
        if (threadIdx.x == 0) {
            stage_begin = clock64();
        }
        __syncthreads();
        ComputeFrontierDistances<kFixedNumChunks>(
            frontier, frontier_slots, num_chunks, query_tables, pq_codes,
            frontier_query_tables_shared, lane_id, warp_id,
            frontier_pq_warps, kFrontierPqModeCurrent, 0U, &profile);
        __syncthreads();
        if (threadIdx.x == 0) {
            profile.pq_cycles += clock64() - stage_begin;
        }
        __syncthreads();
        unsigned long long candidate_sort_cycles_before = 0;
        bool had_full_prefix_before_merge = false;
        if (threadIdx.x == 0) {
            candidate_sort_cycles_before = profile.candidate_sort_cycles;
            had_full_prefix_before_merge = stats.valid_candidates >= top_l;
        }
        __syncthreads();
        std::uint32_t accepted_frontier = 0U;
        std::uint32_t topk_changed = 0U;
        const std::uint32_t merged_valid_count =
            MergeFrontierIntoCandidates(
                candidates, candidate_capacity, stats.valid_candidates,
                frontier, frontier_capacity, out_candidates,
                learned_stop.top_k, &topk_changed, &accepted_frontier,
                &profile);
        if (threadIdx.x == 0) {
            if (topk_changed != 0U) {
                learned_stop_topk_changed = 1U;
            }
            const unsigned long long candidate_sort_delta =
                profile.candidate_sort_cycles - candidate_sort_cycles_before;
            if (had_full_prefix_before_merge) {
                profile.candidate_sort_after_full_prefix_cycles +=
                    candidate_sort_delta;
            } else {
                profile.candidate_sort_before_full_prefix_cycles +=
                    candidate_sort_delta;
            }
            stats.valid_candidates = merged_valid_count;
            if (stats.first_full_prefix_iteration ==
                    kInvalidFullPrefixIteration &&
                merged_valid_count >= top_l) {
                stats.first_full_prefix_iteration = 0U;
            }
            ++hash_iteration;
            stage_begin = clock64();
        }
        __syncthreads();
        const bool rebuild_hash = hash_iteration >= 3U;
        RebuildPrefixState(
            candidates,
            std::min<std::size_t>(stats.valid_candidates, top_l),
            unexpanded_bits, unexpanded_words, rebuild_hash, visited_hash);
        if (threadIdx.x == 0) {
            const unsigned long long delta = clock64() - stage_begin;
            profile.queue_scan_cycles += delta;
            profile.queue_cycles += delta;
            if (rebuild_hash) {
                profile.hash_rebuild_cycles += delta;
                hash_iteration = 0U;
            }
        }
        __syncthreads();
    }

    for (std::size_t i = threadIdx.x; i < candidate_capacity;
         i += blockDim.x) {
        out_candidates[i] = candidates[i];
    }
    if (threadIdx.x == 0) {
        DeviceMicrobatchQueryState state{};
        state.active = stats.valid_candidates != 0U ? 1U : 0U;
        state.hash_iteration = hash_iteration;
        state.learned_stop_topk_changed = learned_stop_topk_changed;
        for (std::size_t i = 0; i < kMaxLearnedStopTopK; ++i) {
            state.learned_stop_prev_topk_ids[i] = kInvalidNodeId;
        }
        query_states[local_query_id] = state;
        out_stats_base[global_query_id] = stats;
        out_profile_base[global_query_id] = profile;
    }
}

__global__ void topology_microbatch_select_kernel(
    const std::uint32_t* topology,
    std::uint64_t topology_cached_node_count,
    std::uint64_t num_nodes,
    std::uint32_t degree,
    std::size_t query_offset,
    std::size_t num_queries,
    std::uint32_t top_l,
    std::uint32_t search_width,
    std::uint32_t max_expansions,
    std::uint32_t candidate_capacity,
    std::uint32_t candidate_stop_prefix,
    std::uint32_t candidate_stop_use_expanded,
    DeviceLearnedStopConfig,
    DeviceCandidate* candidates_base,
    DeviceSearchStats* stats_base,
    DeviceProfileCycles* profile_base,
    DeviceMicrobatchQueryState* query_states,
    std::uint32_t* selected_nodes_base,
    std::uint32_t* response_neighbors,
    float* response_exact_distances,
    DeviceTopologyIoRequest* requests,
    std::uint32_t* request_count,
    std::size_t request_capacity,
    std::uint32_t* active_count) {
    const DeviceLearnedStopConfig& learned_stop = kDeviceTraversalLearnedStopConfig;
    const std::size_t local_query_id = blockIdx.x;
    if (local_query_id >= num_queries) {
        return;
    }
    const std::size_t global_query_id = query_offset + local_query_id;
    extern __shared__ unsigned char shared_memory[];
    auto* candidates = reinterpret_cast<DeviceCandidate*>(shared_memory);
    auto* unexpanded_bits = reinterpret_cast<std::uint32_t*>(
        AlignPointer<std::uint32_t>(candidates + candidate_capacity));
    const std::size_t unexpanded_words = (top_l + 31U) / 32U;
    const std::size_t candidate_stop_words =
        (candidate_stop_prefix + 31U) / 32U;

    __shared__ DeviceSearchStats stats;
    __shared__ DeviceProfileCycles profile;
    __shared__ DeviceMicrobatchQueryState state;
    __shared__ std::uint32_t selected_nodes[kMaxSearchWidth];
    __shared__ std::uint32_t selected_count;
    __shared__ std::uint32_t should_stop;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    DeviceCandidate* global_candidates =
        candidates_base + global_query_id * candidate_capacity;

    for (std::size_t i = threadIdx.x; i < candidate_capacity;
         i += blockDim.x) {
        candidates[i] = global_candidates[i];
    }
    if (threadIdx.x == 0) {
        stats = stats_base[global_query_id];
        profile = profile_base[global_query_id];
        state = query_states[local_query_id];
        selected_count = 0U;
        should_stop = state.active == 0U ? 1U : 0U;
        state.selected_count = 0U;
        state.selected_uncached_mask = 0U;
    }
    __syncthreads();
    if (state.active != 0U) {
        RebuildPrefixState(
            candidates,
            std::min<std::size_t>(stats.valid_candidates, top_l),
            unexpanded_bits, unexpanded_words, false, nullptr);
        unsigned long long queue_begin = 0;
        if (threadIdx.x == 0) {
            queue_begin = clock64();
        }
        __syncthreads();
        MaybeEvaluateLearnedStop(
            learned_stop, candidates, unexpanded_bits, top_l, stats,
            &state.learned_stop_stage, &state.learned_stop_action,
            &state.learned_stop_has_prev, &state.learned_stop_prev_dk,
            &state.learned_stop_prev_boundary,
            state.learned_stop_prev_topk_ids,
            &state.learned_stop_topk_changed, &profile);
        __syncthreads();
        if (threadIdx.x == 0) {
            const bool learned_stop_triggered =
                learned_stop.enabled != 0U &&
                state.learned_stop_action == 1U;
            bool candidate_stop_triggered = false;
            const bool use_candidate_prefix_stop =
                candidate_stop_prefix != 0U &&
                candidate_stop_prefix < top_l;
            if (use_candidate_prefix_stop) {
                if (candidate_stop_use_expanded != 0U) {
                    candidate_stop_triggered =
                        stats.valid_candidates >= top_l &&
                        stats.expanded_nodes >= candidate_stop_prefix;
                } else {
                    const bool candidate_prefix_ready =
                        stats.valid_candidates >= candidate_stop_prefix;
                    const std::size_t active_stop_words =
                        candidate_prefix_ready ? candidate_stop_words
                                               : unexpanded_words;
                    candidate_stop_triggered =
                        !IsAnyBitSet(unexpanded_bits, active_stop_words);
                }
            } else {
                candidate_stop_triggered =
                    !IsAnyBitSet(unexpanded_bits, unexpanded_words);
            }
            should_stop =
                learned_stop_triggered || candidate_stop_triggered ||
                        stats.valid_candidates == 0U ||
                        stats.expanded_nodes >= max_expansions
                    ? 1U
                    : 0U;
            const unsigned long long delta = clock64() - queue_begin;
            profile.queue_scan_cycles += delta;
            profile.queue_cycles += delta;
            if (should_stop == 0U) {
                ++stats.iterations;
                queue_begin = clock64();
                std::uint32_t count = 0U;
                for (std::size_t word_index = 0;
                     word_index < unexpanded_words && count < search_width;
                     ++word_index) {
                    std::uint32_t word = unexpanded_bits[word_index];
                    while (word != 0U && count < search_width) {
                        const std::size_t bit = static_cast<std::size_t>(
                            __ffs(static_cast<int>(word)) - 1);
                        const std::size_t candidate_index =
                            word_index * 32U + bit;
                        word &= word - 1U;
                        if (candidate_index >= candidate_capacity ||
                            candidate_index >= top_l ||
                            candidate_index >= stats.valid_candidates ||
                            !IsValid(candidates[candidate_index]) ||
                            candidates[candidate_index].expanded()) {
                            continue;
                        }
                        candidates[candidate_index].set_expanded(true);
                        selected_nodes[count] =
                            candidates[candidate_index].raw_node_id();
                        ++stats.expanded_nodes;
                        ++count;
                    }
                }
                selected_count = count;
                state.selected_count = count;
                const unsigned long long select_delta =
                    clock64() - queue_begin;
                profile.queue_select_cycles += select_delta;
                profile.queue_cycles += select_delta;
                if (count == 0U) {
                    should_stop = 1U;
                }
            }
            if (should_stop != 0U) {
                state.active = 0U;
                state.selected_count = 0U;
            } else {
                atomicAdd(active_count, 1U);
            }
        }
        __syncthreads();

        if (should_stop == 0U && warp_id < selected_count) {
            const std::uint32_t node_id = selected_nodes[warp_id];
            const std::size_t response_index =
                local_query_id * search_width + warp_id;
            if (lane_id == 0) {
                selected_nodes_base[response_index] = node_id;
                response_exact_distances[response_index] =
                    std::numeric_limits<float>::infinity();
            }
            const bool uncached =
                static_cast<std::uint64_t>(node_id) >=
                topology_cached_node_count;
            if (uncached) {
                if (lane_id == 0) {
                    const std::uint32_t slot = atomicAdd(request_count, 1U);
                    if (slot < request_capacity) {
                        requests[slot] = DeviceTopologyIoRequest{
                            static_cast<std::uint32_t>(global_query_id),
                            static_cast<std::uint32_t>(response_index), node_id,
                            0U};
                    }
                    atomicOr(&state.selected_uncached_mask,
                             1U << static_cast<std::uint32_t>(warp_id));
                    atomicAdd(&stats.topology_io_pages, 1U);
                }
            } else {
                const void* topology_src =
                    topology + static_cast<std::size_t>(node_id) * degree;
                std::uint32_t* destination =
                    response_neighbors + response_index * degree;
                copyFromHostCacheVec(
                    topology_src, destination,
                    static_cast<std::size_t>(degree) * sizeof(std::uint32_t),
                    __activemask());
            }
        }
        __syncthreads();
    }

    for (std::size_t i = threadIdx.x; i < candidate_capacity;
         i += blockDim.x) {
        global_candidates[i] = candidates[i];
    }
    if (threadIdx.x == 0) {
        query_states[local_query_id] = state;
        stats_base[global_query_id] = stats;
        profile_base[global_query_id] = profile;
    }
}

template <std::size_t kFixedNumChunks = kNoFixedNumChunks>
__global__ void topology_microbatch_resume_kernel(
    std::uint64_t num_nodes,
    std::uint32_t degree,
    const std::uint8_t* pq_codes,
    std::size_t num_chunks,
    const float* query_tables_base,
    std::uint32_t query_stride,
    std::size_t query_offset,
    std::size_t num_queries,
    std::uint32_t top_l,
    std::uint32_t search_width,
    std::uint32_t candidate_capacity,
    std::uint32_t frontier_slots,
    std::uint32_t frontier_capacity,
    std::uint32_t frontier_pq_warps,
    DeviceLearnedStopConfig,
    DeviceTraversalExactReuseConfig exact_reuse,
    std::uint32_t* visited_hash_base,
    DeviceCandidate* candidates_base,
    DeviceSearchStats* stats_base,
    DeviceProfileCycles* profile_base,
    DeviceMicrobatchQueryState* query_states,
    const std::uint32_t* selected_nodes_base,
    const std::uint32_t* response_neighbors,
    const float* response_exact_distances) {
    const DeviceLearnedStopConfig& learned_stop = kDeviceTraversalLearnedStopConfig;
    const std::size_t local_query_id = blockIdx.x;
    if (local_query_id >= num_queries) {
        return;
    }
    const std::size_t global_query_id = query_offset + local_query_id;
    extern __shared__ unsigned char shared_memory[];
    auto* candidates = reinterpret_cast<DeviceCandidate*>(shared_memory);
    auto* frontier = candidates + candidate_capacity;
    auto* staged_neighbors = reinterpret_cast<std::uint32_t*>(
        AlignPointer<vectype>(reinterpret_cast<void*>(frontier + frontier_capacity)));
    const std::size_t unexpanded_words = (top_l + 31U) / 32U;
    auto* unexpanded_bits = AlignPointer<std::uint32_t>(
        staged_neighbors + frontier_slots);
    auto* frontier_query_tables_shared = reinterpret_cast<float*>(
        AlignPointer<float4>(unexpanded_bits + unexpanded_words));

    __shared__ DeviceSearchStats stats;
    __shared__ DeviceProfileCycles profile;
    __shared__ DeviceMicrobatchQueryState state;
    __shared__ std::uint32_t selected_nodes[kMaxSearchWidth];
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    const float* query_tables =
        query_tables_base + global_query_id * query_stride;
    std::uint32_t* visited_hash =
        visited_hash_base + local_query_id * kVisitedHashWords;
    DeviceCandidate* global_candidates =
        candidates_base + global_query_id * candidate_capacity;

    for (std::size_t i = threadIdx.x; i < candidate_capacity;
         i += blockDim.x) {
        candidates[i] = global_candidates[i];
    }
    if (threadIdx.x == 0) {
        stats = stats_base[global_query_id];
        profile = profile_base[global_query_id];
        state = query_states[local_query_id];
    }
    __syncthreads();
    if (state.active == 0U || state.selected_count == 0U) {
        return;
    }
    if (threadIdx.x < state.selected_count) {
        selected_nodes[threadIdx.x] =
            selected_nodes_base[local_query_id * search_width + threadIdx.x];
    }
    __syncthreads();

    if (threadIdx.x == 0 && exact_reuse.enabled != 0U &&
        state.selected_uncached_mask != 0U) {
        const unsigned long long insert_begin = clock64();
        for (std::size_t selected_index = 0;
             selected_index < state.selected_count; ++selected_index) {
            if ((state.selected_uncached_mask & (1U << selected_index)) == 0U) {
                continue;
            }
            const std::size_t response_index =
                local_query_id * search_width + selected_index;
            InsertExactReuse(
                exact_reuse, global_query_id, selected_nodes[selected_index],
                response_exact_distances[response_index], &stats);
        }
        profile.exact_reuse_insert_cycles += clock64() - insert_begin;
    }
    __syncthreads();

    ClearCandidates(frontier, frontier_capacity);
    const std::size_t selected_count = state.selected_count;
    const std::size_t neighbor_count =
        selected_count * static_cast<std::size_t>(degree);
    for (std::size_t i = threadIdx.x; i < neighbor_count;
         i += blockDim.x) {
        const std::size_t selected_index = i / degree;
        const std::size_t neighbor_index = i % degree;
        const std::size_t frontier_index =
            selected_index * degree + neighbor_index;
        if (frontier_index >= frontier_capacity) {
            continue;
        }
        const std::size_t response_index =
            local_query_id * search_width + selected_index;
        const std::uint32_t neighbor_id =
            response_neighbors[response_index * degree + neighbor_index];
        if (neighbor_id == kInvalidNodeId || neighbor_id >= num_nodes ||
            VisitedTest(neighbor_id, visited_hash)) {
            continue;
        }
        frontier[frontier_index].set_raw_node_id(neighbor_id, false);
    }
    __syncthreads();
    for (std::size_t i = threadIdx.x; i < neighbor_count;
         i += blockDim.x) {
        if (!IsValid(frontier[i])) {
            continue;
        }
        VisitedAdd(frontier[i].raw_node_id(), visited_hash);
        atomicAdd(&stats.visited_nodes, 1U);
    }
    __syncthreads();

    const std::uint32_t actual_frontier_slots =
        static_cast<std::uint32_t>(selected_count) * degree;
    unsigned long long stage_begin = 0;
    if (threadIdx.x == 0) {
        stage_begin = clock64();
    }
    __syncthreads();
    ComputeFrontierDistances<kFixedNumChunks>(
        frontier, actual_frontier_slots, num_chunks, query_tables, pq_codes,
        frontier_query_tables_shared, lane_id, warp_id, frontier_pq_warps,
        kFrontierPqModeCurrent, 0U, &profile);
    __syncthreads();
    if (threadIdx.x == 0) {
        profile.pq_cycles += clock64() - stage_begin;
    }
    __syncthreads();

    unsigned long long candidate_sort_cycles_before = 0;
    bool had_full_prefix_before_merge = false;
    if (threadIdx.x == 0) {
        candidate_sort_cycles_before = profile.candidate_sort_cycles;
        had_full_prefix_before_merge = stats.valid_candidates >= top_l;
    }
    __syncthreads();
    std::uint32_t accepted_frontier = 0U;
    std::uint32_t topk_changed = 0U;
    const std::uint32_t merged_valid_count =
        MergeFrontierIntoCandidates(
            candidates, candidate_capacity, stats.valid_candidates, frontier,
            frontier_capacity, global_candidates, learned_stop.top_k,
            &topk_changed, &accepted_frontier, &profile);
    if (threadIdx.x == 0) {
        if (topk_changed != 0U) {
            state.learned_stop_topk_changed = 1U;
        }
        const unsigned long long candidate_sort_delta =
            profile.candidate_sort_cycles - candidate_sort_cycles_before;
        if (had_full_prefix_before_merge) {
            profile.candidate_sort_after_full_prefix_cycles +=
                candidate_sort_delta;
        } else {
            profile.candidate_sort_before_full_prefix_cycles +=
                candidate_sort_delta;
        }
        stats.valid_candidates = merged_valid_count;
        if (stats.first_full_prefix_iteration == kInvalidFullPrefixIteration &&
            merged_valid_count >= top_l) {
            stats.first_full_prefix_iteration = stats.iterations;
        }
        ++state.merge_ordinal;
        ++state.hash_iteration;
        stage_begin = clock64();
    }
    __syncthreads();
    const bool rebuild_hash = state.hash_iteration >= 3U;
    RebuildPrefixState(
        candidates, std::min<std::size_t>(stats.valid_candidates, top_l),
        unexpanded_bits, unexpanded_words, rebuild_hash, visited_hash);
    if (threadIdx.x == 0) {
        const unsigned long long delta = clock64() - stage_begin;
        profile.queue_scan_cycles += delta;
        profile.queue_cycles += delta;
        if (rebuild_hash) {
            profile.hash_rebuild_cycles += delta;
            state.hash_iteration = 0U;
        }
        state.selected_count = 0U;
        state.selected_uncached_mask = 0U;
    }
    __syncthreads();

    for (std::size_t i = threadIdx.x; i < candidate_capacity;
         i += blockDim.x) {
        global_candidates[i] = candidates[i];
    }
    if (threadIdx.x == 0) {
        query_states[local_query_id] = state;
        stats_base[global_query_id] = stats;
        profile_base[global_query_id] = profile;
    }
}
template <std::size_t kFixedNumChunks = kNoFixedNumChunks>
__global__ void topology_traversal_kernel(
    const std::uint32_t* topology,
    const void* topology_ssd,
    std::uint64_t topology_cached_node_count,
    std::uint32_t topology_nodes_per_page,
    DeviceTraversalExactReuseConfig exact_reuse,
    std::uint64_t num_nodes,
    std::uint32_t degree,
    const std::uint8_t* pq_codes,
    std::size_t num_chunks,
    const float* query_tables_base,
    std::uint32_t query_stride,
    const std::uint32_t* entry_offsets,
    const std::uint32_t* entry_ids,
    std::uint32_t top_l,
    std::uint32_t search_width,
    std::uint32_t max_expansions,
    std::uint32_t candidate_capacity,
    std::uint32_t frontier_slots,
    std::uint32_t frontier_capacity,
    std::uint32_t frontier_pq_warps,
    std::uint32_t frontier_pq_mode,
    std::uint32_t gustann_tile_chunks,
    std::uint32_t stage_topology_reads,
    std::uint32_t candidate_stop_prefix,
    std::uint32_t candidate_stop_use_expanded,
    DeviceLearnedStopConfig,
    const std::uint32_t* gt_ids_base,
    std::uint32_t gt_topk,
    std::uint32_t* visited_hash_base,
    std::uint32_t* out_gt_hit_masks_base,
    DeviceCandidate* out_candidates_base,
    DeviceSearchStats* out_stats_base,
    DeviceProfileCycles* out_profile_base,
    std::uint32_t* expanded_trace_base,
    std::uint32_t expanded_trace_stride,
    std::uint32_t capture_prefix,
    std::uint32_t max_debug_snapshots,
    std::uint32_t* debug_snapshot_counts_base,
    DeviceTopologyDebugSnapshot* debug_snapshots_base,
    DeviceCandidate* debug_candidate_snapshots_base) {
    const DeviceLearnedStopConfig& learned_stop = kDeviceTraversalLearnedStopConfig;
    extern __shared__ unsigned char shared_memory[];
    auto* candidates = reinterpret_cast<DeviceCandidate*>(shared_memory);
    auto* frontier = candidates + candidate_capacity;
    auto* staged_neighbors = reinterpret_cast<std::uint32_t*>(
        AlignPointer<vectype>(reinterpret_cast<void*>(frontier + frontier_capacity)));
    const std::size_t unexpanded_words = (top_l + 31U) / 32U;
    const std::size_t candidate_stop_words = (candidate_stop_prefix + 31U) / 32U;
    auto* unexpanded_bits = AlignPointer<std::uint32_t>(staged_neighbors + frontier_slots);
    auto* frontier_query_tables_shared = reinterpret_cast<float*>(
        AlignPointer<float4>(unexpanded_bits + unexpanded_words));

    __shared__ DeviceSearchStats stats;
    __shared__ DeviceProfileCycles profile;
    __shared__ std::uint32_t selected_nodes[kMaxSearchWidth];
    __shared__ std::uint32_t selected_count_shared;
    __shared__ std::uint32_t hash_iteration_shared;
    __shared__ std::uint32_t debug_slot_shared;
    __shared__ std::uint32_t merge_ordinal_shared;
    __shared__ std::uint64_t frontier_checksum_shared;
    __shared__ std::uint64_t visited_hash_checksum_shared;
    __shared__ std::uint32_t learned_stop_stage_shared;
    __shared__ std::uint32_t learned_stop_action_shared;
    __shared__ std::uint32_t learned_stop_has_prev_shared;
    __shared__ std::uint32_t learned_stop_topk_changed_shared;
    __shared__ float learned_stop_prev_dk_shared;
    __shared__ float learned_stop_prev_boundary_shared;
    __shared__ std::uint32_t learned_stop_prev_topk_ids_shared[kMaxLearnedStopTopK];
    const std::size_t query_block_id = blockIdx.x;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    const bool control_thread = threadIdx.x < kWarpSize;
    const float* query_tables = query_tables_base + query_block_id * query_stride;
    const std::uint32_t entry_begin = entry_offsets[query_block_id];
    const std::uint32_t entry_end = entry_offsets[query_block_id + 1];
    const std::uint32_t num_entries = entry_end - entry_begin;
    const std::uint32_t* query_entry_ids = entry_ids + entry_begin;
    const std::uint32_t* query_gt_ids =
        (gt_ids_base != nullptr && gt_topk != 0U) ? (gt_ids_base + query_block_id * gt_topk)
                                                  : nullptr;
    std::uint32_t* visited_hash = visited_hash_base + query_block_id * kVisitedHashWords;
    std::uint32_t* query_gt_hit_mask =
        out_gt_hit_masks_base != nullptr ? (out_gt_hit_masks_base + query_block_id) : nullptr;
    DeviceCandidate* out_candidates = out_candidates_base + query_block_id * candidate_capacity;
    DeviceSearchStats* out_stats = out_stats_base + query_block_id;
    DeviceProfileCycles* out_profile = out_profile_base + query_block_id;
    std::uint32_t* query_expanded_trace =
        expanded_trace_base == nullptr
            ? nullptr : expanded_trace_base + query_block_id * expanded_trace_stride;

    ClearVisitedHashWords(visited_hash);
    if (threadIdx.x < kMaxLearnedStopTopK) {
        learned_stop_prev_topk_ids_shared[threadIdx.x] = kInvalidNodeId;
    }
    if (threadIdx.x == 0) {
        stats = DeviceSearchStats{};
        stats.first_full_prefix_iteration = kInvalidFullPrefixIteration;
        profile = DeviceProfileCycles{};
        hash_iteration_shared = 0U;
        merge_ordinal_shared = 0U;
        learned_stop_stage_shared = 0U;
        learned_stop_action_shared = 0U;
        learned_stop_has_prev_shared = 0U;
        learned_stop_topk_changed_shared = 0U;
        learned_stop_prev_dk_shared = std::numeric_limits<float>::infinity();
        learned_stop_prev_boundary_shared = std::numeric_limits<float>::infinity();
        if (query_gt_hit_mask != nullptr) {
            *query_gt_hit_mask = 0U;
        }
    }
    __syncthreads();

    ClearCandidates(candidates, candidate_capacity);
    ClearCandidates(frontier, frontier_capacity);
    ClearBitWords(unexpanded_bits, unexpanded_words);

    const std::uint32_t entry_batches =
        frontier_slots == 0 ? 0U : (num_entries + frontier_slots - 1U) / frontier_slots;
    for (std::uint32_t batch = 0; batch < entry_batches; ++batch) {
        ClearCandidates(frontier, frontier_capacity);
        for (std::size_t i = threadIdx.x; i < frontier_slots; i += blockDim.x) {
            const std::size_t entry_index = batch * frontier_slots + i;
            if (entry_index >= num_entries) {
                continue;
            }
            const std::uint32_t node_id = query_entry_ids[entry_index];
            if (node_id >= num_nodes || VisitedTest(node_id, visited_hash)) {
                continue;
            }
            frontier[i].set_raw_node_id(node_id, false);
            VisitedAdd(node_id, visited_hash);
            MarkGroundTruthVisited(node_id, query_gt_ids, gt_topk, query_gt_hit_mask);
            atomicAdd(&stats.visited_nodes, 1U);
        }
        __syncthreads();
        unsigned long long stage_begin = 0;
        if (threadIdx.x == 0) {
            stage_begin = clock64();
        }
        __syncthreads();
        ComputeFrontierDistances<kFixedNumChunks>(frontier, frontier_slots, num_chunks,
                                                  query_tables, pq_codes,
                                                  frontier_query_tables_shared, lane_id, warp_id,
                                                  frontier_pq_warps, frontier_pq_mode,
                                                  gustann_tile_chunks, &profile);
        __syncthreads();
        if (threadIdx.x == 0) {
            if (max_debug_snapshots != 0U) {
                const unsigned long long checksum_begin = clock64();
                frontier_checksum_shared = ChecksumCandidates(frontier, frontier_capacity);
                visited_hash_checksum_shared = ChecksumWords(visited_hash, kVisitedHashWords);
                profile.pq_checksum_cycles += clock64() - checksum_begin;
            } else {
                frontier_checksum_shared = 0U;
                visited_hash_checksum_shared = 0U;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            profile.pq_cycles += clock64() - stage_begin;
        }
        __syncthreads();
        const std::uint32_t frontier_valid =
            static_cast<std::uint32_t>(CountValidCandidates(frontier, frontier_capacity));
        unsigned long long candidate_sort_cycles_before = 0;
        bool had_full_prefix_before_merge = false;
        if (threadIdx.x == 0) {
            candidate_sort_cycles_before = profile.candidate_sort_cycles;
            had_full_prefix_before_merge = stats.valid_candidates >= top_l;
        }
        __syncthreads();
        std::uint32_t accepted_frontier = 0U;
        std::uint32_t topk_changed = 0U;
        const std::uint32_t merged_valid_count =
            MergeFrontierIntoCandidates(candidates, candidate_capacity, stats.valid_candidates,
                                        frontier, frontier_capacity, out_candidates,
                                        learned_stop.top_k, &topk_changed,
                                        &accepted_frontier, &profile);
        if (threadIdx.x == 0) {
            if (topk_changed != 0U) {
                learned_stop_topk_changed_shared = 1U;
            }
            const unsigned long long candidate_sort_delta =
                profile.candidate_sort_cycles - candidate_sort_cycles_before;
            if (had_full_prefix_before_merge) {
                profile.candidate_sort_after_full_prefix_cycles += candidate_sort_delta;
            } else {
                profile.candidate_sort_before_full_prefix_cycles += candidate_sort_delta;
            }
            stats.valid_candidates = merged_valid_count;
            if (stats.first_full_prefix_iteration == kInvalidFullPrefixIteration &&
                merged_valid_count >= top_l) {
                stats.first_full_prefix_iteration = 0U;
            }
            ++merge_ordinal_shared;
        }
        __syncthreads();
        CaptureDebugSnapshot(query_block_id, candidates, capture_prefix, merge_ordinal_shared, 0U,
                             stats.iterations, frontier_valid, accepted_frontier, 0U,
                             frontier_checksum_shared, visited_hash_checksum_shared, stats,
                             &debug_slot_shared, debug_snapshot_counts_base,
                             debug_snapshots_base, debug_candidate_snapshots_base,
                             max_debug_snapshots);
        if (threadIdx.x == 0) {
            ++hash_iteration_shared;
            stage_begin = clock64();
        }
        __syncthreads();
        const bool rebuild_hash = hash_iteration_shared >= 3U;
        RebuildPrefixState(candidates, std::min<std::size_t>(stats.valid_candidates, top_l),
                           unexpanded_bits, unexpanded_words, rebuild_hash, visited_hash);
        if (threadIdx.x == 0) {
            const unsigned long long delta = clock64() - stage_begin;
            profile.queue_scan_cycles += delta;
            profile.queue_cycles += delta;
            if (rebuild_hash) {
                profile.hash_rebuild_cycles += delta;
                hash_iteration_shared = 0U;
            }
        }
        __syncthreads();
    }

    while (stats.expanded_nodes < max_expansions) {
        unsigned long long queue_begin = 0;
        if (threadIdx.x == 0) {
            queue_begin = clock64();
        }
        __syncthreads();

        if (stats.valid_candidates == 0) {
            break;
        }
        MaybeEvaluateLearnedStop(learned_stop, candidates, unexpanded_bits, top_l, stats,
                                 &learned_stop_stage_shared, &learned_stop_action_shared,
                                 &learned_stop_has_prev_shared, &learned_stop_prev_dk_shared,
                                 &learned_stop_prev_boundary_shared,
                                 learned_stop_prev_topk_ids_shared,
                                 &learned_stop_topk_changed_shared, &profile);
        __syncthreads();
        if (learned_stop.enabled != 0U && learned_stop_action_shared == 1U) {
            if (threadIdx.x == 0) {
                const unsigned long long delta = clock64() - queue_begin;
                profile.queue_scan_cycles += delta;
                profile.queue_cycles += delta;
            }
            break;
        }
        const bool use_candidate_prefix_stop =
            candidate_stop_prefix != 0U && candidate_stop_prefix < top_l;
        bool candidate_stop_triggered = false;
        if (use_candidate_prefix_stop) {
            if (candidate_stop_use_expanded != 0U) {
                candidate_stop_triggered =
                    stats.valid_candidates >= top_l &&
                    stats.expanded_nodes >= candidate_stop_prefix;
            } else {
                const bool candidate_prefix_ready =
                    stats.valid_candidates >= candidate_stop_prefix;
                const std::size_t active_stop_words =
                    candidate_prefix_ready ? candidate_stop_words : unexpanded_words;
                candidate_stop_triggered = !IsAnyBitSet(unexpanded_bits, active_stop_words);
            }
        } else {
            candidate_stop_triggered = !IsAnyBitSet(unexpanded_bits, unexpanded_words);
        }
        if (candidate_stop_triggered) {
            if (threadIdx.x == 0) {
                const unsigned long long delta = clock64() - queue_begin;
                profile.queue_scan_cycles += delta;
                profile.queue_cycles += delta;
            }
            break;
        }
        if (threadIdx.x == 0) {
            const unsigned long long delta = clock64() - queue_begin;
            profile.queue_scan_cycles += delta;
            profile.queue_cycles += delta;
            ++stats.iterations;
        }
        __syncthreads();

        ClearCandidates(frontier, frontier_capacity);
        if (threadIdx.x == 0) {
            selected_count_shared = 0U;
            queue_begin = clock64();
        }
        __syncthreads();
        if (control_thread) {
            if (lane_id == 0) {
                std::uint32_t selected_count = 0;
                for (std::size_t word_index = 0;
                     word_index < unexpanded_words && selected_count < search_width;
                     ++word_index) {
                    std::uint32_t word = unexpanded_bits[word_index];
                    while (word != 0U && selected_count < search_width) {
                        const std::size_t bit =
                            static_cast<std::size_t>(__ffs(static_cast<int>(word)) - 1);
                        const std::size_t candidate_index = word_index * 32U + bit;
                        word &= (word - 1U);
                        unexpanded_bits[word_index] &= ~(1U << bit);
                        if (candidate_index >= candidate_capacity ||
                            candidate_index >= static_cast<std::size_t>(top_l) ||
                            candidate_index >= static_cast<std::size_t>(stats.valid_candidates) ||
                            !IsValid(candidates[candidate_index]) ||
                            candidates[candidate_index].expanded()) {
                            continue;
                        }
                        candidates[candidate_index].set_expanded(true);
                        const std::uint32_t expanded_node_id =
                            candidates[candidate_index].raw_node_id();
                        selected_nodes[selected_count] = expanded_node_id;
                        if (query_expanded_trace != nullptr &&
                            stats.expanded_nodes < expanded_trace_stride) {
                            query_expanded_trace[stats.expanded_nodes] = expanded_node_id;
                        }
                        ++stats.expanded_nodes;
                        ++selected_count;
                    }
                }
                selected_count_shared = selected_count;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned long long delta = clock64() - queue_begin;
            profile.queue_select_cycles += delta;
            profile.queue_cycles += delta;
        }

        const std::uint32_t selected_count = selected_count_shared;

        if (selected_count == 0) {
            break;
        }

        if (stage_topology_reads != 0U && warp_id < selected_count) {
            const std::uint32_t current_node = selected_nodes[warp_id];
            if (lane_id == 0 &&
                (topology_ssd != nullptr || exact_reuse.enabled != 0U) &&
                static_cast<std::uint64_t>(current_node) >= topology_cached_node_count) {
                atomicAdd(&stats.topology_io_pages, 1U);
            }
            auto* shared_dst = staged_neighbors + static_cast<std::size_t>(warp_id) * degree;
            CopyTopologyNeighborsForWarp(
                topology, topology_ssd, topology_cached_node_count,
                topology_nodes_per_page, num_nodes, degree, current_node,
                query_block_id, exact_reuse, &stats, &profile, shared_dst,
                static_cast<std::uint32_t>(lane_id), __activemask());
        }
        __syncthreads();

        for (std::size_t i = threadIdx.x; i < static_cast<std::size_t>(selected_count) * degree;
             i += blockDim.x) {
            const std::size_t selected_index = i / degree;
            const std::size_t neighbor_index = i % degree;
            const std::uint32_t current_node = selected_nodes[selected_index];
            const std::size_t frontier_index = selected_index * degree + neighbor_index;
            if (frontier_index >= frontier_capacity) {
                continue;
            }
            const std::uint32_t neighbor_id =
                stage_topology_reads != 0U
                    ? staged_neighbors[selected_index * degree + neighbor_index]
                    : topology[static_cast<std::size_t>(current_node) * degree + neighbor_index];
            if (neighbor_id == kInvalidNodeId || neighbor_id >= num_nodes ||
                VisitedTest(neighbor_id, visited_hash)) {
                continue;
            }
            frontier[frontier_index].set_raw_node_id(neighbor_id, false);
            VisitedAdd(neighbor_id, visited_hash);
            MarkGroundTruthVisited(neighbor_id, query_gt_ids, gt_topk, query_gt_hit_mask);
            atomicAdd(&stats.visited_nodes, 1U);
        }
        __syncthreads();

        const std::uint32_t actual_frontier_slots = selected_count * degree;
        unsigned long long stage_begin = 0;
        if (threadIdx.x == 0) {
            stage_begin = clock64();
        }
        __syncthreads();
        ComputeFrontierDistances<kFixedNumChunks>(frontier, actual_frontier_slots, num_chunks,
                                                  query_tables, pq_codes,
                                                  frontier_query_tables_shared, lane_id, warp_id,
                                                  frontier_pq_warps, frontier_pq_mode,
                                                  gustann_tile_chunks, &profile);
        __syncthreads();
        if (threadIdx.x == 0) {
            if (max_debug_snapshots != 0U) {
                const unsigned long long checksum_begin = clock64();
                frontier_checksum_shared = ChecksumCandidates(frontier, frontier_capacity);
                visited_hash_checksum_shared = ChecksumWords(visited_hash, kVisitedHashWords);
                profile.pq_checksum_cycles += clock64() - checksum_begin;
            } else {
                frontier_checksum_shared = 0U;
                visited_hash_checksum_shared = 0U;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            profile.pq_cycles += clock64() - stage_begin;
        }
        __syncthreads();
        const std::uint32_t frontier_valid =
            static_cast<std::uint32_t>(CountValidCandidates(frontier, frontier_capacity));
        unsigned long long candidate_sort_cycles_before = 0;
        bool had_full_prefix_before_merge = false;
        if (threadIdx.x == 0) {
            candidate_sort_cycles_before = profile.candidate_sort_cycles;
            had_full_prefix_before_merge = stats.valid_candidates >= top_l;
        }
        __syncthreads();
        std::uint32_t accepted_frontier = 0U;
        std::uint32_t topk_changed = 0U;
        const std::uint32_t merged_valid_count =
            MergeFrontierIntoCandidates(candidates, candidate_capacity, stats.valid_candidates,
                                        frontier, frontier_capacity, out_candidates,
                                        learned_stop.top_k, &topk_changed,
                                        &accepted_frontier, &profile);
        if (threadIdx.x == 0) {
            if (topk_changed != 0U) {
                learned_stop_topk_changed_shared = 1U;
            }
            const unsigned long long candidate_sort_delta =
                profile.candidate_sort_cycles - candidate_sort_cycles_before;
            if (had_full_prefix_before_merge) {
                profile.candidate_sort_after_full_prefix_cycles += candidate_sort_delta;
            } else {
                profile.candidate_sort_before_full_prefix_cycles += candidate_sort_delta;
            }
            stats.valid_candidates = merged_valid_count;
            if (stats.first_full_prefix_iteration == kInvalidFullPrefixIteration &&
                merged_valid_count >= top_l) {
                stats.first_full_prefix_iteration = stats.iterations;
            }
            ++merge_ordinal_shared;
        }
        __syncthreads();
        CaptureDebugSnapshot(query_block_id, candidates, capture_prefix, merge_ordinal_shared, 1U,
                             stats.iterations, frontier_valid, accepted_frontier, selected_count,
                             frontier_checksum_shared, visited_hash_checksum_shared, stats,
                             &debug_slot_shared, debug_snapshot_counts_base,
                             debug_snapshots_base, debug_candidate_snapshots_base,
                             max_debug_snapshots);
        if (threadIdx.x == 0) {
            ++hash_iteration_shared;
            stage_begin = clock64();
        }
        __syncthreads();

        const bool rebuild_hash = hash_iteration_shared >= 3U;
        RebuildPrefixState(candidates, std::min<std::size_t>(stats.valid_candidates, top_l),
                           unexpanded_bits, unexpanded_words, rebuild_hash, visited_hash);
        if (threadIdx.x == 0) {
            const unsigned long long delta = clock64() - stage_begin;
            profile.queue_scan_cycles += delta;
            profile.queue_cycles += delta;
            if (rebuild_hash) {
                profile.hash_rebuild_cycles += delta;
                hash_iteration_shared = 0U;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_stats[0] = stats;
        out_profile[0] = profile;
    }
    for (std::size_t i = threadIdx.x; i < candidate_capacity; i += blockDim.x) {
        out_candidates[i] = candidates[i];
    }
}

template <std::size_t kFixedNumChunks = kNoFixedNumChunks>
__global__ void topology_traversal_kernel_high_l(
    const std::uint32_t* topology,
    const void* topology_ssd,
    std::uint64_t topology_cached_node_count,
    std::uint32_t topology_nodes_per_page,
    DeviceTraversalExactReuseConfig exact_reuse,
    std::uint64_t num_nodes,
    std::uint32_t degree,
    const std::uint8_t* pq_codes,
    std::size_t num_chunks,
    const float* query_tables_base,
    std::uint32_t query_stride,
    const std::uint32_t* entry_offsets,
    const std::uint32_t* entry_ids,
    std::uint32_t top_l,
    std::uint32_t search_width,
    std::uint32_t max_expansions,
    std::uint32_t candidate_capacity,
    std::uint32_t frontier_slots,
    std::uint32_t frontier_capacity,
    std::uint32_t frontier_pq_warps,
    std::uint32_t frontier_pq_mode,
    std::uint32_t gustann_tile_chunks,
    std::uint32_t stage_topology_reads,
    std::uint32_t candidate_stop_prefix,
    std::uint32_t candidate_stop_use_expanded,
    DeviceLearnedStopConfig,
    const std::uint32_t* gt_ids_base,
    std::uint32_t gt_topk,
    std::uint32_t* visited_hash_base,
    std::uint32_t* out_gt_hit_masks_base,
    DeviceCandidate* out_candidates_base,
    DeviceSearchStats* out_stats_base,
    DeviceProfileCycles* out_profile_base,
    std::uint32_t capture_prefix,
    std::uint32_t max_debug_snapshots,
    std::uint32_t* debug_snapshot_counts_base,
    DeviceTopologyDebugSnapshot* debug_snapshots_base,
    DeviceCandidate* debug_candidate_snapshots_base) {
    const DeviceLearnedStopConfig& learned_stop = kDeviceTraversalLearnedStopConfig;
    extern __shared__ unsigned char shared_memory[];
    auto* candidates = reinterpret_cast<DeviceCandidate*>(shared_memory);
    auto* frontier = candidates + candidate_capacity;
    auto* staged_neighbors = reinterpret_cast<std::uint32_t*>(
        AlignPointer<vectype>(reinterpret_cast<void*>(frontier + frontier_capacity)));
    const std::size_t unexpanded_words = (top_l + 31U) / 32U;
    const std::size_t candidate_stop_words = (candidate_stop_prefix + 31U) / 32U;
    auto* unexpanded_bits = AlignPointer<std::uint32_t>(staged_neighbors + frontier_slots);
    auto* frontier_output_positions = unexpanded_bits + unexpanded_words;
    auto* frontier_output_count = frontier_output_positions + frontier_capacity;
    auto* frontier_query_tables_shared = reinterpret_cast<float*>(
        AlignPointer<float4>(frontier_output_count + 1U));

    __shared__ DeviceSearchStats stats;
    __shared__ DeviceProfileCycles profile;
    __shared__ std::uint32_t selected_nodes[kMaxSearchWidth];
    __shared__ std::uint32_t selected_count_shared;
    __shared__ std::uint32_t hash_iteration_shared;
    __shared__ std::uint32_t debug_slot_shared;
    __shared__ std::uint32_t merge_ordinal_shared;
    __shared__ std::uint64_t frontier_checksum_shared;
    __shared__ std::uint64_t visited_hash_checksum_shared;
    __shared__ std::uint32_t learned_stop_stage_shared;
    __shared__ std::uint32_t learned_stop_action_shared;
    __shared__ std::uint32_t learned_stop_has_prev_shared;
    __shared__ std::uint32_t learned_stop_topk_changed_shared;
    __shared__ float learned_stop_prev_dk_shared;
    __shared__ float learned_stop_prev_boundary_shared;
    __shared__ std::uint32_t learned_stop_prev_topk_ids_shared[kMaxLearnedStopTopK];
    const std::size_t query_block_id = blockIdx.x;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    const bool control_thread = threadIdx.x < kWarpSize;
    const float* query_tables = query_tables_base + query_block_id * query_stride;
    const std::uint32_t entry_begin = entry_offsets[query_block_id];
    const std::uint32_t entry_end = entry_offsets[query_block_id + 1];
    const std::uint32_t num_entries = entry_end - entry_begin;
    const std::uint32_t* query_entry_ids = entry_ids + entry_begin;
    const std::uint32_t* query_gt_ids =
        (gt_ids_base != nullptr && gt_topk != 0U) ? (gt_ids_base + query_block_id * gt_topk)
                                                  : nullptr;
    std::uint32_t* visited_hash = visited_hash_base + query_block_id * kVisitedHashWords;
    std::uint32_t* query_gt_hit_mask =
        out_gt_hit_masks_base != nullptr ? (out_gt_hit_masks_base + query_block_id) : nullptr;
    DeviceCandidate* out_candidates = out_candidates_base + query_block_id * candidate_capacity;
    DeviceSearchStats* out_stats = out_stats_base + query_block_id;
    DeviceProfileCycles* out_profile = out_profile_base + query_block_id;

    ClearVisitedHashWords(visited_hash);
    if (threadIdx.x < kMaxLearnedStopTopK) {
        learned_stop_prev_topk_ids_shared[threadIdx.x] = kInvalidNodeId;
    }
    if (threadIdx.x == 0) {
        stats = DeviceSearchStats{};
        stats.first_full_prefix_iteration = kInvalidFullPrefixIteration;
        profile = DeviceProfileCycles{};
        hash_iteration_shared = 0U;
        merge_ordinal_shared = 0U;
        learned_stop_stage_shared = 0U;
        learned_stop_action_shared = 0U;
        learned_stop_has_prev_shared = 0U;
        learned_stop_topk_changed_shared = 0U;
        learned_stop_prev_dk_shared = std::numeric_limits<float>::infinity();
        learned_stop_prev_boundary_shared = std::numeric_limits<float>::infinity();
        if (query_gt_hit_mask != nullptr) {
            *query_gt_hit_mask = 0U;
        }
    }
    __syncthreads();

    ClearCandidates(candidates, candidate_capacity);
    ClearCandidates(frontier, frontier_capacity);
    ClearBitWords(unexpanded_bits, unexpanded_words);

    const std::uint32_t entry_batches =
        frontier_slots == 0 ? 0U : (num_entries + frontier_slots - 1U) / frontier_slots;
    for (std::uint32_t batch = 0; batch < entry_batches; ++batch) {
        ClearCandidates(frontier, frontier_capacity);
        for (std::size_t i = threadIdx.x; i < frontier_slots; i += blockDim.x) {
            const std::size_t entry_index = batch * frontier_slots + i;
            if (entry_index >= num_entries) {
                continue;
            }
            const std::uint32_t node_id = query_entry_ids[entry_index];
            if (node_id >= num_nodes || VisitedTest(node_id, visited_hash)) {
                continue;
            }
            frontier[i].set_raw_node_id(node_id, false);
            VisitedAdd(node_id, visited_hash);
            MarkGroundTruthVisited(node_id, query_gt_ids, gt_topk, query_gt_hit_mask);
            atomicAdd(&stats.visited_nodes, 1U);
        }
        __syncthreads();
        unsigned long long stage_begin = 0;
        if (threadIdx.x == 0) {
            stage_begin = clock64();
        }
        __syncthreads();
        ComputeFrontierDistances<kFixedNumChunks>(frontier, frontier_slots, num_chunks,
                                                  query_tables, pq_codes,
                                                  frontier_query_tables_shared, lane_id, warp_id,
                                                  frontier_pq_warps, frontier_pq_mode,
                                                  gustann_tile_chunks, &profile);
        __syncthreads();
        if (threadIdx.x == 0) {
            if (max_debug_snapshots != 0U) {
                const unsigned long long checksum_begin = clock64();
                frontier_checksum_shared = ChecksumCandidates(frontier, frontier_capacity);
                visited_hash_checksum_shared = ChecksumWords(visited_hash, kVisitedHashWords);
                profile.pq_checksum_cycles += clock64() - checksum_begin;
            } else {
                frontier_checksum_shared = 0U;
                visited_hash_checksum_shared = 0U;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            profile.pq_cycles += clock64() - stage_begin;
        }
        __syncthreads();
        const std::uint32_t frontier_valid =
            static_cast<std::uint32_t>(CountValidCandidates(frontier, frontier_capacity));
        unsigned long long candidate_sort_cycles_before = 0;
        bool had_full_prefix_before_merge = false;
        if (threadIdx.x == 0) {
            candidate_sort_cycles_before = profile.candidate_sort_cycles;
            had_full_prefix_before_merge = stats.valid_candidates >= top_l;
        }
        __syncthreads();
        std::uint32_t accepted_frontier = 0U;
        std::uint32_t topk_changed = 0U;
        const std::uint32_t merged_valid_count = MergeFrontierIntoCandidatesHighL(
            candidates, candidate_capacity, stats.valid_candidates, frontier, frontier_capacity,
            out_candidates, frontier_output_positions, frontier_output_count,
            learned_stop.top_k, &topk_changed,
            &accepted_frontier, &profile);
        if (threadIdx.x == 0) {
            if (topk_changed != 0U) {
                learned_stop_topk_changed_shared = 1U;
            }
            const unsigned long long candidate_sort_delta =
                profile.candidate_sort_cycles - candidate_sort_cycles_before;
            if (had_full_prefix_before_merge) {
                profile.candidate_sort_after_full_prefix_cycles += candidate_sort_delta;
            } else {
                profile.candidate_sort_before_full_prefix_cycles += candidate_sort_delta;
            }
            stats.valid_candidates = merged_valid_count;
            if (stats.first_full_prefix_iteration == kInvalidFullPrefixIteration &&
                merged_valid_count >= top_l) {
                stats.first_full_prefix_iteration = 0U;
            }
            ++merge_ordinal_shared;
        }
        __syncthreads();
        CaptureDebugSnapshot(query_block_id, candidates, capture_prefix, merge_ordinal_shared, 0U,
                             stats.iterations, frontier_valid, accepted_frontier, 0U,
                             frontier_checksum_shared, visited_hash_checksum_shared, stats,
                             &debug_slot_shared, debug_snapshot_counts_base,
                             debug_snapshots_base, debug_candidate_snapshots_base,
                             max_debug_snapshots);
        if (threadIdx.x == 0) {
            ++hash_iteration_shared;
            stage_begin = clock64();
        }
        __syncthreads();
        const bool rebuild_hash = hash_iteration_shared >= 3U;
        RebuildPrefixState(candidates, std::min<std::size_t>(stats.valid_candidates, top_l),
                           unexpanded_bits, unexpanded_words, rebuild_hash, visited_hash);
        if (threadIdx.x == 0) {
            const unsigned long long delta = clock64() - stage_begin;
            profile.queue_scan_cycles += delta;
            profile.queue_cycles += delta;
            if (rebuild_hash) {
                profile.hash_rebuild_cycles += delta;
                hash_iteration_shared = 0U;
            }
        }
        __syncthreads();
    }

    while (stats.expanded_nodes < max_expansions) {
        unsigned long long queue_begin = 0;
        if (threadIdx.x == 0) {
            queue_begin = clock64();
        }
        __syncthreads();

        if (stats.valid_candidates == 0) {
            break;
        }
        MaybeEvaluateLearnedStop(learned_stop, candidates, unexpanded_bits, top_l, stats,
                                 &learned_stop_stage_shared, &learned_stop_action_shared,
                                 &learned_stop_has_prev_shared, &learned_stop_prev_dk_shared,
                                 &learned_stop_prev_boundary_shared,
                                 learned_stop_prev_topk_ids_shared,
                                 &learned_stop_topk_changed_shared, &profile);
        __syncthreads();
        if (learned_stop.enabled != 0U && learned_stop_action_shared == 1U) {
            if (threadIdx.x == 0) {
                const unsigned long long delta = clock64() - queue_begin;
                profile.queue_scan_cycles += delta;
                profile.queue_cycles += delta;
            }
            break;
        }
        const bool use_candidate_prefix_stop =
            candidate_stop_prefix != 0U && candidate_stop_prefix < top_l;
        bool candidate_stop_triggered = false;
        if (use_candidate_prefix_stop) {
            if (candidate_stop_use_expanded != 0U) {
                candidate_stop_triggered =
                    stats.valid_candidates >= top_l &&
                    stats.expanded_nodes >= candidate_stop_prefix;
            } else {
                const bool candidate_prefix_ready =
                    stats.valid_candidates >= candidate_stop_prefix;
                const std::size_t active_stop_words =
                    candidate_prefix_ready ? candidate_stop_words : unexpanded_words;
                candidate_stop_triggered = !IsAnyBitSet(unexpanded_bits, active_stop_words);
            }
        } else {
            candidate_stop_triggered = !IsAnyBitSet(unexpanded_bits, unexpanded_words);
        }
        if (candidate_stop_triggered) {
            if (threadIdx.x == 0) {
                const unsigned long long delta = clock64() - queue_begin;
                profile.queue_scan_cycles += delta;
                profile.queue_cycles += delta;
            }
            break;
        }
        if (threadIdx.x == 0) {
            const unsigned long long delta = clock64() - queue_begin;
            profile.queue_scan_cycles += delta;
            profile.queue_cycles += delta;
            ++stats.iterations;
        }
        __syncthreads();

        ClearCandidates(frontier, frontier_capacity);
        if (threadIdx.x == 0) {
            selected_count_shared = 0U;
            queue_begin = clock64();
        }
        __syncthreads();
        if (control_thread) {
            if (lane_id == 0) {
                std::uint32_t selected_count = 0;
                for (std::size_t word_index = 0;
                     word_index < unexpanded_words && selected_count < search_width;
                     ++word_index) {
                    std::uint32_t word = unexpanded_bits[word_index];
                    while (word != 0U && selected_count < search_width) {
                        const std::size_t bit =
                            static_cast<std::size_t>(__ffs(static_cast<int>(word)) - 1);
                        const std::size_t candidate_index = word_index * 32U + bit;
                        word &= (word - 1U);
                        unexpanded_bits[word_index] &= ~(1U << bit);
                        if (candidate_index >= candidate_capacity ||
                            candidate_index >= static_cast<std::size_t>(top_l) ||
                            candidate_index >= static_cast<std::size_t>(stats.valid_candidates) ||
                            !IsValid(candidates[candidate_index]) ||
                            candidates[candidate_index].expanded()) {
                            continue;
                        }
                        candidates[candidate_index].set_expanded(true);
                        selected_nodes[selected_count] =
                            candidates[candidate_index].raw_node_id();
                        ++stats.expanded_nodes;
                        ++selected_count;
                    }
                }
                selected_count_shared = selected_count;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            const unsigned long long delta = clock64() - queue_begin;
            profile.queue_select_cycles += delta;
            profile.queue_cycles += delta;
        }

        const std::uint32_t selected_count = selected_count_shared;

        if (selected_count == 0) {
            break;
        }

        if (stage_topology_reads != 0U && warp_id < selected_count) {
            const std::uint32_t current_node = selected_nodes[warp_id];
            if (lane_id == 0 &&
                (topology_ssd != nullptr || exact_reuse.enabled != 0U) &&
                static_cast<std::uint64_t>(current_node) >= topology_cached_node_count) {
                atomicAdd(&stats.topology_io_pages, 1U);
            }
            auto* shared_dst = staged_neighbors + static_cast<std::size_t>(warp_id) * degree;
            CopyTopologyNeighborsForWarp(
                topology, topology_ssd, topology_cached_node_count,
                topology_nodes_per_page, num_nodes, degree, current_node,
                query_block_id, exact_reuse, &stats, &profile, shared_dst,
                static_cast<std::uint32_t>(lane_id), __activemask());
        }
        __syncthreads();

        for (std::size_t i = threadIdx.x; i < static_cast<std::size_t>(selected_count) * degree;
             i += blockDim.x) {
            const std::size_t selected_index = i / degree;
            const std::size_t neighbor_index = i % degree;
            const std::uint32_t current_node = selected_nodes[selected_index];
            const std::size_t frontier_index = selected_index * degree + neighbor_index;
            if (frontier_index >= frontier_capacity) {
                continue;
            }
            const std::uint32_t neighbor_id =
                stage_topology_reads != 0U
                    ? staged_neighbors[selected_index * degree + neighbor_index]
                    : topology[static_cast<std::size_t>(current_node) * degree + neighbor_index];
            if (neighbor_id == kInvalidNodeId || neighbor_id >= num_nodes ||
                VisitedTest(neighbor_id, visited_hash)) {
                continue;
            }
            frontier[frontier_index].set_raw_node_id(neighbor_id, false);
            VisitedAdd(neighbor_id, visited_hash);
            MarkGroundTruthVisited(neighbor_id, query_gt_ids, gt_topk, query_gt_hit_mask);
            atomicAdd(&stats.visited_nodes, 1U);
        }
        __syncthreads();

        const std::uint32_t actual_frontier_slots = selected_count * degree;
        unsigned long long stage_begin = 0;
        if (threadIdx.x == 0) {
            stage_begin = clock64();
        }
        __syncthreads();
        ComputeFrontierDistances<kFixedNumChunks>(frontier, actual_frontier_slots, num_chunks,
                                                  query_tables, pq_codes,
                                                  frontier_query_tables_shared, lane_id, warp_id,
                                                  frontier_pq_warps, frontier_pq_mode,
                                                  gustann_tile_chunks, &profile);
        __syncthreads();
        if (threadIdx.x == 0) {
            if (max_debug_snapshots != 0U) {
                const unsigned long long checksum_begin = clock64();
                frontier_checksum_shared = ChecksumCandidates(frontier, frontier_capacity);
                visited_hash_checksum_shared = ChecksumWords(visited_hash, kVisitedHashWords);
                profile.pq_checksum_cycles += clock64() - checksum_begin;
            } else {
                frontier_checksum_shared = 0U;
                visited_hash_checksum_shared = 0U;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            profile.pq_cycles += clock64() - stage_begin;
        }
        __syncthreads();
        const std::uint32_t frontier_valid =
            static_cast<std::uint32_t>(CountValidCandidates(frontier, frontier_capacity));
        unsigned long long candidate_sort_cycles_before = 0;
        bool had_full_prefix_before_merge = false;
        if (threadIdx.x == 0) {
            candidate_sort_cycles_before = profile.candidate_sort_cycles;
            had_full_prefix_before_merge = stats.valid_candidates >= top_l;
        }
        __syncthreads();
        std::uint32_t accepted_frontier = 0U;
        std::uint32_t topk_changed = 0U;
        const std::uint32_t merged_valid_count = MergeFrontierIntoCandidatesHighL(
            candidates, candidate_capacity, stats.valid_candidates, frontier, frontier_capacity,
            out_candidates, frontier_output_positions, frontier_output_count,
            learned_stop.top_k, &topk_changed,
            &accepted_frontier, &profile);
        if (threadIdx.x == 0) {
            if (topk_changed != 0U) {
                learned_stop_topk_changed_shared = 1U;
            }
            const unsigned long long candidate_sort_delta =
                profile.candidate_sort_cycles - candidate_sort_cycles_before;
            if (had_full_prefix_before_merge) {
                profile.candidate_sort_after_full_prefix_cycles += candidate_sort_delta;
            } else {
                profile.candidate_sort_before_full_prefix_cycles += candidate_sort_delta;
            }
            stats.valid_candidates = merged_valid_count;
            if (stats.first_full_prefix_iteration == kInvalidFullPrefixIteration &&
                merged_valid_count >= top_l) {
                stats.first_full_prefix_iteration = stats.iterations;
            }
            ++merge_ordinal_shared;
        }
        __syncthreads();
        CaptureDebugSnapshot(query_block_id, candidates, capture_prefix, merge_ordinal_shared, 1U,
                             stats.iterations, frontier_valid, accepted_frontier, selected_count,
                             frontier_checksum_shared, visited_hash_checksum_shared, stats,
                             &debug_slot_shared, debug_snapshot_counts_base,
                             debug_snapshots_base, debug_candidate_snapshots_base,
                             max_debug_snapshots);
        if (threadIdx.x == 0) {
            ++hash_iteration_shared;
            stage_begin = clock64();
        }
        __syncthreads();

        const bool rebuild_hash = hash_iteration_shared >= 3U;
        RebuildPrefixState(candidates, std::min<std::size_t>(stats.valid_candidates, top_l),
                           unexpanded_bits, unexpanded_words, rebuild_hash, visited_hash);
        if (threadIdx.x == 0) {
            const unsigned long long delta = clock64() - stage_begin;
            profile.queue_scan_cycles += delta;
            profile.queue_cycles += delta;
            if (rebuild_hash) {
                profile.hash_rebuild_cycles += delta;
                hash_iteration_shared = 0U;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_stats[0] = stats;
        out_profile[0] = profile;
    }
    for (std::size_t i = threadIdx.x; i < candidate_capacity; i += blockDim.x) {
        out_candidates[i] = candidates[i];
    }
}

std::vector<RankedCandidate> BuildSortedCandidates(const std::vector<DeviceCandidate>& host_candidates,
                                                   std::size_t candidate_limit) {
    std::vector<RankedCandidate> result;
    result.reserve(candidate_limit);
    for (const DeviceCandidate& candidate : host_candidates) {
        if (!IsValid(candidate)) {
            continue;
        }
        result.push_back(RankedCandidate{
            candidate.distance,
            candidate.raw_node_id(),
            candidate.expanded(),
        });
        if (result.size() == candidate_limit) {
            break;
        }
    }
    return result;
}

std::vector<RankedCandidate> BuildTopK(const std::vector<RankedCandidate>& sorted_candidates,
                                       std::size_t top_k) {
    std::vector<RankedCandidate> topk;
    topk.reserve(top_k);
    for (const RankedCandidate& candidate : sorted_candidates) {
        if (!candidate.valid()) {
            continue;
        }
        topk.push_back(candidate);
        if (topk.size() == top_k) {
            break;
        }
    }
    return topk;
}

}  // namespace

DeviceTopologyBatchResult RunTopologySearchKernelBatchDeviceImpl(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const CudaBuffer<std::uint32_t>& entry_offsets_buffer,
    const CudaBuffer<std::uint32_t>& entry_ids_buffer,
    const CudaBuffer<std::uint32_t>* gt_ids_buffer,
    std::uint32_t gt_topk,
    std::size_t query_offset,
    std::size_t num_queries,
    const TopologySearchParams& params,
    const DeviceTopologyDebugConfig* debug_config,
    DeviceTopologyDebugTrace* out_debug_trace,
    TopologyLaunchHostCallback launch_callback,
    void* launch_callback_context,
    cudaStream_t stream) {
    DeviceTopologyBatchResult result;
    result.num_queries = num_queries;
    if (num_queries == 0) {
        return result;
    }
    if (params.search_width > kMaxSearchWidth) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatch", "search_width must be at most 32."));
    }
    if (query_offset + num_queries > distance_oracle.num_queries()) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatch", "query range is out of bounds."));
    }
    const std::uint32_t degree = resources.degree();
    if (resources.num_nodes() >= static_cast<std::uint64_t>(kRawNodeIdMask)) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatch",
            "topology kernel packed node_id requires num_nodes < 2^31 - 1."));
    }
    const std::size_t candidate_limit =
        std::max(params.top_l, params.candidate_queue_size == 0 ? params.top_l
                                                                : params.candidate_queue_size);
    const std::size_t candidate_capacity = NextPowerOfTwo(candidate_limit);
    result.candidate_capacity = candidate_capacity;
    const std::size_t frontier_slots = params.search_width * degree;
    const std::size_t frontier_capacity = NextPowerOfTwo(frontier_slots == 0 ? 1 : frontier_slots);
    const std::size_t unexpanded_words = (params.top_l + 31U) / 32U;
    const std::size_t shared_bytes_base =
        (candidate_capacity + frontier_capacity) * sizeof(DeviceCandidate) +
        frontier_slots * sizeof(std::uint32_t) + unexpanded_words * sizeof(std::uint32_t) +
        alignof(vectype);
    const std::size_t high_l_shared_bytes_base =
        shared_bytes_base + (frontier_capacity + 1U) * sizeof(std::uint32_t);

    CudaBuffer<std::uint32_t> visited_hash =
        CudaBuffer<std::uint32_t>::Allocate(num_queries * kVisitedHashWords);
    result.candidate_buffer =
        CudaBuffer<DeviceCandidate>::Allocate(num_queries * candidate_capacity);
    result.stats_buffer = CudaBuffer<DeviceSearchStats>::Allocate(num_queries);
    result.profile_buffer = CudaBuffer<DeviceProfileCycles>::Allocate(num_queries);
    if (params.enable_expanded_trace) {
        if (params.max_expansions == 0 ||
            params.max_expansions > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("expanded trace requires a uint32 max_expansions value.");
        }
        result.expanded_trace_stride = params.max_expansions;
        result.expanded_trace_buffer = CudaBuffer<std::uint32_t>::Allocate(
            num_queries * result.expanded_trace_stride);
    }
    const bool enable_debug =
        debug_config != nullptr && out_debug_trace != nullptr && debug_config->max_snapshots != 0U &&
        debug_config->capture_prefix != 0U;
    CudaBuffer<std::uint32_t> debug_snapshot_counts;
    CudaBuffer<DeviceTopologyDebugSnapshot> debug_snapshots;
    CudaBuffer<DeviceCandidate> debug_candidate_snapshots;
    std::uint32_t capture_prefix = 0U;
    std::uint32_t max_debug_snapshots = 0U;
    if (enable_debug) {
        capture_prefix = std::min<std::uint32_t>(static_cast<std::uint32_t>(candidate_capacity),
                                                 debug_config->capture_prefix);
        max_debug_snapshots = debug_config->max_snapshots;
        debug_snapshot_counts = CudaBuffer<std::uint32_t>::Allocate(num_queries);
        debug_snapshots = CudaBuffer<DeviceTopologyDebugSnapshot>::Allocate(
            num_queries * static_cast<std::size_t>(max_debug_snapshots));
        debug_candidate_snapshots = CudaBuffer<DeviceCandidate>::Allocate(
            num_queries * static_cast<std::size_t>(max_debug_snapshots) * capture_prefix);
        ThrowIfCudaError(cudaMemsetAsync(debug_snapshot_counts.get(), 0,
                                         debug_snapshot_counts.size() * sizeof(std::uint32_t),
                                         stream),
                         "cudaMemsetAsync");
    }
    if (gt_ids_buffer != nullptr && gt_topk != 0U) {
        if (gt_topk > 32U) {
            throw std::runtime_error(BuildErrorMessage(
                "RunTopologySearchKernelBatch", "gt_topk must be at most 32."));
        }
        if (gt_ids_buffer->size() != num_queries * static_cast<std::size_t>(gt_topk)) {
            throw std::runtime_error(BuildErrorMessage(
                "RunTopologySearchKernelBatch",
                "gt id buffer size must equal num_queries * gt_topk."));
        }
        result.gt_hit_mask_buffer = CudaBuffer<std::uint32_t>::Allocate(num_queries);
        ThrowIfCudaError(cudaMemsetAsync(result.gt_hit_mask_buffer.get(), 0,
                                         result.gt_hit_mask_buffer.size() * sizeof(std::uint32_t),
                                         stream),
                         "cudaMemsetAsync");
    }

    ThrowIfCudaError(cudaMemsetAsync(visited_hash.get(), 0,
                                     visited_hash.size() * sizeof(std::uint32_t), stream),
                     "cudaMemsetAsync");

    const std::size_t query_stride =
        distance_oracle.pq_index().host().num_chunks * kNumPqCentroids;
    const float* query_tables =
        distance_oracle.query_tables().device_tables().get() + query_offset * query_stride;
    std::uint32_t stage_topology_reads = 1U;
    if (const char* env = std::getenv("TOPOANNS_STAGE_TOPOLOGY_READ")) {
        if (env[0] == '0') {
            stage_topology_reads = 0U;
        }
    }
    const void* topology_ssd_array = resources.topology_device_read_handle();
    const void* combined_ssd_array = resources.combined_node_device_read_handle();
    const std::uint64_t topology_cached_node_count =
        std::min<std::uint64_t>(params.topology_cached_node_count, resources.num_nodes());
    DeviceTraversalExactReuseConfig exact_reuse{};
    const bool exact_reuse_enabled =
        params.enable_exact_reuse && topology_cached_node_count < resources.num_nodes();
    if (exact_reuse_enabled) {
        if (combined_ssd_array == nullptr || params.exact_reuse_device_queries == nullptr) {
            throw std::runtime_error(BuildErrorMessage(
                "RunTopologySearchKernelBatch",
                "exact reuse requires combined-node SSD and device query buffers."));
        }
        if (params.exact_reuse_query_dim != 96 && params.exact_reuse_query_dim != 128 &&
            params.exact_reuse_query_dim != 512 && params.exact_reuse_query_dim != 768) {
            throw std::runtime_error(BuildErrorMessage(
                "RunTopologySearchKernelBatch", "unsupported exact reuse query dimension."));
        }
        const std::size_t vector_bytes = params.exact_reuse_query_dim * sizeof(float);
        const std::size_t minimum_node_bytes =
            vector_bytes + (static_cast<std::size_t>(degree) + 1U) * sizeof(std::uint32_t);
        if (params.combined_node_bytes < minimum_node_bytes ||
            params.combined_nodes_per_page == 0 ||
            static_cast<std::size_t>(params.combined_node_bytes) *
                    params.combined_nodes_per_page >
                kDefaultPageSizeBytes) {
            throw std::runtime_error(BuildErrorMessage(
                "RunTopologySearchKernelBatch", "invalid combined-node record layout."));
        }
        std::size_t cache_capacity = params.exact_reuse_cache_capacity;
        if (cache_capacity == 0) {
            cache_capacity = NextPowerOfTwo(std::max<std::size_t>(
                2U, static_cast<std::size_t>(params.max_expansions) * 2U));
        }
        if ((cache_capacity & (cache_capacity - 1U)) != 0 ||
            cache_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(BuildErrorMessage(
                "RunTopologySearchKernelBatch",
                "exact reuse cache capacity must be a power of two."));
        }
        result.exact_reuse_cache_capacity = cache_capacity;
        result.exact_reuse_node_ids =
            CudaBuffer<std::uint32_t>::Allocate(num_queries * cache_capacity);
        result.exact_reuse_distances =
            CudaBuffer<float>::Allocate(num_queries * cache_capacity);
        ThrowIfCudaError(cudaMemsetAsync(
                             result.exact_reuse_node_ids.get(), 0xff,
                             result.exact_reuse_node_ids.size() * sizeof(std::uint32_t), stream),
                         "cudaMemsetAsync(exact_reuse_node_ids)");
        exact_reuse.enabled = 1U;
        exact_reuse.query_dim = static_cast<std::uint32_t>(params.exact_reuse_query_dim);
        exact_reuse.vector_bytes = static_cast<std::uint32_t>(vector_bytes);
        exact_reuse.node_bytes = params.combined_node_bytes;
        exact_reuse.nodes_per_page = params.combined_nodes_per_page;
        exact_reuse.cache_capacity = static_cast<std::uint32_t>(cache_capacity);
        exact_reuse.combined_ssd = combined_ssd_array;
        exact_reuse.queries = params.exact_reuse_device_queries;
        exact_reuse.cache_node_ids = result.exact_reuse_node_ids.get();
        exact_reuse.cache_distances = result.exact_reuse_distances.get();
    }
    const bool topology_ssd_enabled =
        topology_ssd_array != nullptr && topology_cached_node_count < resources.num_nodes();
    if (topology_cached_node_count < resources.num_nodes() &&
        topology_ssd_array == nullptr && !exact_reuse_enabled) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatch",
            "partial topology cache requires topology-only or combined-node SSD provider."));
    }
    const std::size_t topology_node_bytes =
        static_cast<std::size_t>(degree) * sizeof(std::uint32_t);
    if (topology_ssd_enabled && (topology_node_bytes == 0 ||
                                 kDefaultPageSizeBytes % topology_node_bytes != 0)) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatch",
            "topology SSD path requires fixed-size nodes that divide the 4KB page size."));
    }
    const std::uint32_t topology_nodes_per_page =
        static_cast<std::uint32_t>(kDefaultPageSizeBytes / topology_node_bytes);
    if (topology_ssd_enabled || exact_reuse.enabled != 0U) {
        stage_topology_reads = 1U;
    }
    bool force_low_l_kernel = false;
    if (const char* env = std::getenv("TOPOANNS_FORCE_LOW_L_KERNEL")) {
        force_low_l_kernel = env[0] == '1';
    }
    bool enable_high_l_kernel = false;
    if (const char* env = std::getenv("TOPOANNS_ENABLE_HIGH_L_KERNEL")) {
        enable_high_l_kernel = env[0] == '1';
    }
    std::uint32_t high_l_kernel_min_top_l = 2048U;
    if (const char* env = std::getenv("TOPOANNS_HIGH_L_KERNEL_MIN_TOP_L")) {
        const auto parsed = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
        if (parsed != 0U) {
            high_l_kernel_min_top_l = parsed;
        }
    }
    bool disable_pq_num_chunks_specialization = false;
    if (const char* env = std::getenv("TOPOANNS_DISABLE_PQ_NUMCHUNKS_SPECIALIZATION")) {
        disable_pq_num_chunks_specialization = env[0] == '1';
    }
    std::uint32_t frontier_pq_warps = 1U;
    if (const char* env = std::getenv("TOPOANNS_FRONTIER_PQ_WARPS")) {
        const auto parsed = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
        if (parsed != 0U) {
            frontier_pq_warps = parsed;
        }
    }
    frontier_pq_warps =
        std::min<std::uint32_t>(frontier_pq_warps,
                                static_cast<std::uint32_t>(kTopologyBlockThreads / kWarpSize));
    std::uint32_t frontier_pq_mode = kFrontierPqModeCurrent;
    if (const char* env = std::getenv("TOPOANNS_FRONTIER_PQ_MODE")) {
        const std::string value = TrimAscii(env);
        if (value == "bang") {
            frontier_pq_mode = kFrontierPqModeBang;
        } else if (value == "gustann") {
            frontier_pq_mode = kFrontierPqModeGustann;
        }
    }
    std::uint32_t gustann_tile_chunks = 0U;
    if (const char* env = std::getenv("TOPOANNS_GUSTANN_LUT_TILE_CHUNKS")) {
        const auto parsed = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
        if (parsed >= 2U && parsed <= kMaxFrontierPqTileChunks) {
            gustann_tile_chunks = parsed;
        }
    }
    const std::size_t gustann_tile_shared_bytes =
        frontier_pq_mode == kFrontierPqModeGustann && gustann_tile_chunks != 0U
            ? alignof(float4) +
                  2U * static_cast<std::size_t>(gustann_tile_chunks) * kNumPqCentroids *
                      sizeof(float)
            : 0U;
    const std::size_t shared_bytes = shared_bytes_base + gustann_tile_shared_bytes;
    const std::size_t high_l_shared_bytes =
        high_l_shared_bytes_base + gustann_tile_shared_bytes;
    int current_device = 0;
    ThrowIfCudaError(cudaGetDevice(&current_device), "cudaGetDevice");
    int max_optin_shared_bytes = 0;
    ThrowIfCudaError(
        cudaDeviceGetAttribute(&max_optin_shared_bytes,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               current_device),
        "cudaDeviceGetAttribute(cudaDevAttrMaxSharedMemoryPerBlockOptin)");
    const std::size_t shared_memory_hard_limit =
        static_cast<std::size_t>(std::max(0, max_optin_shared_bytes));
    const bool launches_high_l_kernel =
        enable_high_l_kernel && !force_low_l_kernel &&
        params.top_l >= high_l_kernel_min_top_l;
    const std::size_t selected_shared_bytes =
        launches_high_l_kernel ? high_l_shared_bytes : shared_bytes;
    if (selected_shared_bytes > shared_memory_hard_limit) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatch",
            "shared memory requirement is too large for selected frontier PQ tile."));
    }
    std::uint32_t candidate_stop_divisor = 1U;
    if (const char* env = std::getenv("TOPOANNS_CANDIDATE_STOP_DIVISOR")) {
        const auto parsed = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
        if (parsed > 1U) {
            candidate_stop_divisor = parsed;
        }
    }
    std::uint32_t candidate_stop_min_prefix = 32U;
    if (const char* env = std::getenv("TOPOANNS_CANDIDATE_STOP_MIN_PREFIX")) {
        const auto parsed = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
        if (parsed != 0U) {
            candidate_stop_min_prefix = parsed;
        }
    }
    std::uint32_t candidate_stop_prefix = static_cast<std::uint32_t>(params.top_l);
    if (candidate_stop_divisor > 1U) {
        candidate_stop_prefix =
            std::max(candidate_stop_min_prefix,
                     static_cast<std::uint32_t>(params.top_l / candidate_stop_divisor));
        candidate_stop_prefix =
            std::min(candidate_stop_prefix, static_cast<std::uint32_t>(params.top_l));
    }
    if (const char* env = std::getenv("TOPOANNS_CANDIDATE_STOP_PREFIX")) {
        const auto parsed = static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
        if (parsed != 0U) {
            candidate_stop_prefix =
                std::min(parsed, static_cast<std::uint32_t>(params.top_l));
        }
    }
    std::uint32_t candidate_stop_use_expanded = 0U;
    if (const char* env = std::getenv("TOPOANNS_CANDIDATE_STOP_USE_EXPANDED")) {
        const std::string value = TrimAscii(env);
        if (!(value.empty() || value == "0" || value == "false" || value == "FALSE")) {
            candidate_stop_use_expanded = 1U;
        }
    }
    DeviceLearnedStopConfig learned_stop{};
    if (const char* env = std::getenv("TOPOANNS_LEARNED_STOP_MODEL")) {
        learned_stop = LoadLearnedStopConfigFromFile(
            env, static_cast<std::uint32_t>(params.top_l),
            static_cast<std::uint32_t>(params.top_k));
    }
    ThrowIfCudaError(
        cudaMemcpyToSymbol(kDeviceTraversalLearnedStopConfig, &learned_stop,
                           sizeof(learned_stop)),
        "cudaMemcpyToSymbol(kDeviceTraversalLearnedStopConfig)");
    const bool use_specialized_search_pq_kernel =
        !disable_pq_num_chunks_specialization &&
        distance_oracle.pq_index().host().num_chunks == kSpecializedSearchPqNumChunks;

    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    if (launch_callback != nullptr) {
        launch_callback(launch_callback_context);
    }
    ThrowIfCudaError(cudaEventRecord(kernel_begin, stream), "cudaEventRecord");
    if (params.enable_expanded_trace && enable_high_l_kernel && !force_low_l_kernel &&
        params.top_l >= high_l_kernel_min_top_l) {
        throw std::runtime_error(
            "expanded trace is supported only by the low-L topology kernel.");
    }
    if (enable_high_l_kernel && !force_low_l_kernel &&
        params.top_l >= high_l_kernel_min_top_l) {
        if (use_specialized_search_pq_kernel) {
            ConfigureKernelDynamicSharedMemory(
                topology_traversal_kernel_high_l<kSpecializedSearchPqNumChunks>,
                high_l_shared_bytes);
            PopulateKernelOccupancy(
                topology_traversal_kernel_high_l<kSpecializedSearchPqNumChunks>,
                high_l_shared_bytes, params.search_width, num_queries,
                params.enable_occupancy_profile, &result);
            topology_traversal_kernel_high_l<kSpecializedSearchPqNumChunks>
                <<<num_queries, kTopologyBlockThreads, high_l_shared_bytes, stream>>>(
                    resources.device_topology_data(),
                    topology_ssd_array,
                    topology_cached_node_count,
                    topology_nodes_per_page,
                    exact_reuse,
                    resources.num_nodes(),
                    degree,
                    distance_oracle.pq_index().device_codes(),
                    distance_oracle.pq_index().host().num_chunks,
                    query_tables,
                    static_cast<std::uint32_t>(query_stride),
                    entry_offsets_buffer.get(),
                    entry_ids_buffer.get(),
                    static_cast<std::uint32_t>(params.top_l),
                    static_cast<std::uint32_t>(params.search_width),
                    static_cast<std::uint32_t>(params.max_expansions),
                    static_cast<std::uint32_t>(candidate_capacity),
                    static_cast<std::uint32_t>(frontier_slots),
                    static_cast<std::uint32_t>(frontier_capacity),
                    frontier_pq_warps,
                    frontier_pq_mode,
                    gustann_tile_chunks,
                    stage_topology_reads,
                    candidate_stop_prefix,
                    candidate_stop_use_expanded,
                    learned_stop,
                    gt_ids_buffer != nullptr ? gt_ids_buffer->get() : nullptr,
                    gt_topk,
                    visited_hash.get(),
                    result.gt_hit_mask_buffer.empty() ? nullptr : result.gt_hit_mask_buffer.get(),
                    result.candidate_buffer.get(),
                    result.stats_buffer.get(),
                    result.profile_buffer.get(),
                    capture_prefix,
                    max_debug_snapshots,
                    enable_debug ? debug_snapshot_counts.get() : nullptr,
                    enable_debug ? debug_snapshots.get() : nullptr,
                    enable_debug ? debug_candidate_snapshots.get() : nullptr);
        } else {
            ConfigureKernelDynamicSharedMemory(topology_traversal_kernel_high_l<kNoFixedNumChunks>,
                                               high_l_shared_bytes);
            PopulateKernelOccupancy(
                topology_traversal_kernel_high_l<kNoFixedNumChunks>,
                high_l_shared_bytes, params.search_width, num_queries,
                params.enable_occupancy_profile, &result);
            topology_traversal_kernel_high_l<kNoFixedNumChunks>
                <<<num_queries, kTopologyBlockThreads, high_l_shared_bytes, stream>>>(
                    resources.device_topology_data(),
                    topology_ssd_array,
                    topology_cached_node_count,
                    topology_nodes_per_page,
                    exact_reuse,
                    resources.num_nodes(),
                    degree,
                    distance_oracle.pq_index().device_codes(),
                    distance_oracle.pq_index().host().num_chunks,
                    query_tables,
                    static_cast<std::uint32_t>(query_stride),
                    entry_offsets_buffer.get(),
                    entry_ids_buffer.get(),
                    static_cast<std::uint32_t>(params.top_l),
                    static_cast<std::uint32_t>(params.search_width),
                    static_cast<std::uint32_t>(params.max_expansions),
                    static_cast<std::uint32_t>(candidate_capacity),
                    static_cast<std::uint32_t>(frontier_slots),
                    static_cast<std::uint32_t>(frontier_capacity),
                    frontier_pq_warps,
                    frontier_pq_mode,
                    gustann_tile_chunks,
                    stage_topology_reads,
                    candidate_stop_prefix,
                    candidate_stop_use_expanded,
                    learned_stop,
                    gt_ids_buffer != nullptr ? gt_ids_buffer->get() : nullptr,
                    gt_topk,
                    visited_hash.get(),
                    result.gt_hit_mask_buffer.empty() ? nullptr : result.gt_hit_mask_buffer.get(),
                    result.candidate_buffer.get(),
                    result.stats_buffer.get(),
                    result.profile_buffer.get(),
                    capture_prefix,
                    max_debug_snapshots,
                    enable_debug ? debug_snapshot_counts.get() : nullptr,
                    enable_debug ? debug_snapshots.get() : nullptr,
                    enable_debug ? debug_candidate_snapshots.get() : nullptr);
        }
        ThrowIfCudaError(cudaGetLastError(), "topology_traversal_kernel_high_l");
    } else {
        if (use_specialized_search_pq_kernel) {
            ConfigureKernelDynamicSharedMemory(
                topology_traversal_kernel<kSpecializedSearchPqNumChunks>, shared_bytes);
            PopulateKernelOccupancy(
                topology_traversal_kernel<kSpecializedSearchPqNumChunks>, shared_bytes,
                params.search_width, num_queries, params.enable_occupancy_profile,
                &result);
            topology_traversal_kernel<kSpecializedSearchPqNumChunks>
                <<<num_queries, kTopologyBlockThreads, shared_bytes, stream>>>(
                    resources.device_topology_data(),
                    topology_ssd_array,
                    topology_cached_node_count,
                    topology_nodes_per_page,
                    exact_reuse,
                    resources.num_nodes(),
                    degree,
                    distance_oracle.pq_index().device_codes(),
                    distance_oracle.pq_index().host().num_chunks,
                    query_tables,
                    static_cast<std::uint32_t>(query_stride),
                    entry_offsets_buffer.get(),
                    entry_ids_buffer.get(),
                    static_cast<std::uint32_t>(params.top_l),
                    static_cast<std::uint32_t>(params.search_width),
                    static_cast<std::uint32_t>(params.max_expansions),
                    static_cast<std::uint32_t>(candidate_capacity),
                    static_cast<std::uint32_t>(frontier_slots),
                    static_cast<std::uint32_t>(frontier_capacity),
                    frontier_pq_warps,
                    frontier_pq_mode,
                    gustann_tile_chunks,
                    stage_topology_reads,
                    candidate_stop_prefix,
                    candidate_stop_use_expanded,
                    learned_stop,
                    gt_ids_buffer != nullptr ? gt_ids_buffer->get() : nullptr,
                    gt_topk,
                    visited_hash.get(),
                    result.gt_hit_mask_buffer.empty() ? nullptr : result.gt_hit_mask_buffer.get(),
                    result.candidate_buffer.get(),
                    result.stats_buffer.get(),
                    result.profile_buffer.get(),
                    result.expanded_trace_buffer.empty() ? nullptr : result.expanded_trace_buffer.get(),
                    static_cast<std::uint32_t>(result.expanded_trace_stride),
                    capture_prefix,
                    max_debug_snapshots,
                    enable_debug ? debug_snapshot_counts.get() : nullptr,
                    enable_debug ? debug_snapshots.get() : nullptr,
                    enable_debug ? debug_candidate_snapshots.get() : nullptr);
        } else {
            ConfigureKernelDynamicSharedMemory(topology_traversal_kernel<kNoFixedNumChunks>,
                                               shared_bytes);
            PopulateKernelOccupancy(
                topology_traversal_kernel<kNoFixedNumChunks>, shared_bytes,
                params.search_width, num_queries, params.enable_occupancy_profile,
                &result);
            topology_traversal_kernel<kNoFixedNumChunks>
                <<<num_queries, kTopologyBlockThreads, shared_bytes, stream>>>(
                    resources.device_topology_data(),
                    topology_ssd_array,
                    topology_cached_node_count,
                    topology_nodes_per_page,
                    exact_reuse,
                    resources.num_nodes(),
                    degree,
                    distance_oracle.pq_index().device_codes(),
                    distance_oracle.pq_index().host().num_chunks,
                    query_tables,
                    static_cast<std::uint32_t>(query_stride),
                    entry_offsets_buffer.get(),
                    entry_ids_buffer.get(),
                    static_cast<std::uint32_t>(params.top_l),
                    static_cast<std::uint32_t>(params.search_width),
                    static_cast<std::uint32_t>(params.max_expansions),
                    static_cast<std::uint32_t>(candidate_capacity),
                    static_cast<std::uint32_t>(frontier_slots),
                    static_cast<std::uint32_t>(frontier_capacity),
                    frontier_pq_warps,
                    frontier_pq_mode,
                    gustann_tile_chunks,
                    stage_topology_reads,
                    candidate_stop_prefix,
                    candidate_stop_use_expanded,
                    learned_stop,
                    gt_ids_buffer != nullptr ? gt_ids_buffer->get() : nullptr,
                    gt_topk,
                    visited_hash.get(),
                    result.gt_hit_mask_buffer.empty() ? nullptr : result.gt_hit_mask_buffer.get(),
                    result.candidate_buffer.get(),
                    result.stats_buffer.get(),
                    result.profile_buffer.get(),
                    result.expanded_trace_buffer.empty() ? nullptr : result.expanded_trace_buffer.get(),
                    static_cast<std::uint32_t>(result.expanded_trace_stride),
                    capture_prefix,
                    max_debug_snapshots,
                    enable_debug ? debug_snapshot_counts.get() : nullptr,
                    enable_debug ? debug_snapshots.get() : nullptr,
                    enable_debug ? debug_candidate_snapshots.get() : nullptr);
        }
        ThrowIfCudaError(cudaGetLastError(), "topology_traversal_kernel");
    }
    ThrowIfCudaError(cudaEventRecord(kernel_end, stream), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);
    result.kernel_ms = static_cast<double>(kernel_ms);
    if (enable_debug) {
        out_debug_trace->num_queries = num_queries;
        out_debug_trace->max_snapshots = max_debug_snapshots;
        out_debug_trace->capture_prefix = capture_prefix;
        out_debug_trace->snapshot_counts = debug_snapshot_counts.CopyToHost();
        out_debug_trace->snapshots = debug_snapshots.CopyToHost();
        out_debug_trace->candidate_snapshots = debug_candidate_snapshots.CopyToHost();
    }
    return result;
}

std::vector<TopologySearchResult> MaterializeTopologyResults(
    DeviceTopologyBatchResult&& device_result,
    const TopologySearchParams& params,
    TopologySearchBatchProfile* out_profile) {
    std::vector<TopologySearchResult> empty_results(device_result.num_queries);
    if (device_result.num_queries == 0) {
        return empty_results;
    }
    const std::size_t candidate_limit =
        std::max(params.top_l, params.candidate_queue_size == 0 ? params.top_l
                                                                : params.candidate_queue_size);
    const auto candidate_download_begin = std::chrono::steady_clock::now();
    const std::vector<DeviceCandidate> host_candidates = device_result.candidate_buffer.CopyToHost();
    const auto candidate_download_end = std::chrono::steady_clock::now();
    const auto stats_download_begin = std::chrono::steady_clock::now();
    const std::vector<DeviceSearchStats> host_stats = device_result.stats_buffer.CopyToHost();
    const auto stats_download_end = std::chrono::steady_clock::now();
    const std::vector<DeviceProfileCycles> host_profile = device_result.profile_buffer.CopyToHost();

    std::vector<TopologySearchResult> results(device_result.num_queries);
    const auto postprocess_begin = std::chrono::steady_clock::now();
    double candidate_materialize_ms = 0.0;
    double topology_topk_extract_ms = 0.0;
    for (std::size_t query_id = 0; query_id < device_result.num_queries; ++query_id) {
        const auto begin =
            host_candidates.begin() +
            static_cast<std::ptrdiff_t>(query_id * device_result.candidate_capacity);
        const auto end = begin + static_cast<std::ptrdiff_t>(device_result.candidate_capacity);
        const std::vector<DeviceCandidate> query_candidates(begin, end);
        const auto candidate_materialize_begin = std::chrono::steady_clock::now();
        results[query_id].sorted_candidates =
            BuildSortedCandidates(query_candidates, candidate_limit);
        const auto candidate_materialize_end = std::chrono::steady_clock::now();
        const auto topology_topk_begin = std::chrono::steady_clock::now();
        results[query_id].topk =
            BuildTopK(results[query_id].sorted_candidates, params.top_k);
        const auto topology_topk_end = std::chrono::steady_clock::now();
        results[query_id].stats.visited_nodes = host_stats[query_id].visited_nodes;
        results[query_id].stats.expanded_nodes = host_stats[query_id].expanded_nodes;
        results[query_id].stats.topology_io_pages = host_stats[query_id].topology_io_pages;
        results[query_id].stats.iterations = host_stats[query_id].iterations;
        candidate_materialize_ms +=
            std::chrono::duration<double, std::milli>(candidate_materialize_end -
                                                      candidate_materialize_begin)
                .count();
        topology_topk_extract_ms +=
            std::chrono::duration<double, std::milli>(topology_topk_end - topology_topk_begin)
                .count();
    }
    const auto postprocess_end = std::chrono::steady_clock::now();

    if (out_profile != nullptr) {
        int device = 0;
        ThrowIfCudaError(cudaGetDevice(&device), "cudaGetDevice");
        cudaDeviceProp props;
        ThrowIfCudaError(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
        double pq_ms = 0.0;
        double queue_ms = 0.0;
        const double cycles_per_ms = static_cast<double>(props.clockRate);
        for (const DeviceProfileCycles& query_profile : host_profile) {
            pq_ms += static_cast<double>(query_profile.pq_cycles) / cycles_per_ms;
            queue_ms += static_cast<double>(query_profile.queue_cycles) / cycles_per_ms;
            out_profile->sum_query_pq_compute_ms +=
                static_cast<double>(query_profile.pq_compute_cycles) / cycles_per_ms;
            out_profile->sum_query_pq_prefetch_issue_ms +=
                static_cast<double>(query_profile.pq_prefetch_issue_cycles) / cycles_per_ms;
            out_profile->sum_query_pq_prefetch_wait_ms +=
                static_cast<double>(query_profile.pq_prefetch_wait_cycles) / cycles_per_ms;
            out_profile->sum_query_pq_checksum_ms +=
                static_cast<double>(query_profile.pq_checksum_cycles) / cycles_per_ms;
            out_profile->sum_query_queue_scan_ms +=
                static_cast<double>(query_profile.queue_scan_cycles) / cycles_per_ms;
            out_profile->sum_query_queue_select_ms +=
                static_cast<double>(query_profile.queue_select_cycles) / cycles_per_ms;
            out_profile->sum_query_frontier_sort_ms +=
                static_cast<double>(query_profile.frontier_sort_cycles) / cycles_per_ms;
            out_profile->sum_query_tail_merge_ms +=
                static_cast<double>(query_profile.tail_merge_cycles) / cycles_per_ms;
            out_profile->sum_query_candidate_sort_ms +=
                static_cast<double>(query_profile.candidate_sort_cycles) / cycles_per_ms;
            out_profile->sum_query_hash_rebuild_ms +=
                static_cast<double>(query_profile.hash_rebuild_cycles) / cycles_per_ms;
            out_profile->sum_query_learned_stop_model_ms +=
                static_cast<double>(query_profile.learned_stop_model_cycles) / cycles_per_ms;
            out_profile->sum_query_learned_stop_feature_ms +=
                static_cast<double>(query_profile.learned_stop_feature_cycles) / cycles_per_ms;
            out_profile->sum_query_learned_stop_find_first_set_ms +=
                static_cast<double>(query_profile.learned_stop_find_first_set_cycles) / cycles_per_ms;
            out_profile->sum_query_learned_stop_count_bits_ms +=
                static_cast<double>(query_profile.learned_stop_count_bits_cycles) / cycles_per_ms;
            out_profile->sum_query_learned_stop_topk_churn_ms +=
                static_cast<double>(query_profile.learned_stop_topk_churn_cycles) / cycles_per_ms;
            out_profile->sum_query_learned_stop_logit_eval_ms +=
                static_cast<double>(query_profile.learned_stop_logit_eval_cycles) / cycles_per_ms;
        }
        out_profile->kernel_ms += device_result.kernel_ms;
        out_profile->candidate_download_ms +=
            std::chrono::duration<double, std::milli>(candidate_download_end -
                                                      candidate_download_begin)
                .count();
        out_profile->stats_download_ms +=
            std::chrono::duration<double, std::milli>(stats_download_end - stats_download_begin)
                .count();
        out_profile->host_postprocess_ms +=
            std::chrono::duration<double, std::milli>(postprocess_end - postprocess_begin)
                .count();
        out_profile->candidate_materialize_ms += candidate_materialize_ms;
        out_profile->topology_topk_extract_ms += topology_topk_extract_ms;
        out_profile->sum_query_pq_distance_ms += pq_ms;
        out_profile->sum_query_queue_update_ms += queue_ms;
    }
    return results;
}

std::vector<TopologySearchResult> RunTopologySearchKernelBatch(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const std::vector<std::vector<std::uint32_t>>& entries_by_query,
    std::size_t query_offset,
    std::size_t num_queries,
    const TopologySearchParams& params,
    TopologySearchBatchProfile* out_profile) {
    if (entries_by_query.size() < num_queries) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatch", "entries_by_query is smaller than num_queries."));
    }

    std::vector<std::uint32_t> host_entry_offsets(num_queries + 1, 0U);
    std::vector<std::uint32_t> host_entry_ids;
    for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
        host_entry_offsets[query_id + 1] =
            host_entry_offsets[query_id] +
            static_cast<std::uint32_t>(entries_by_query[query_id].size());
        host_entry_ids.insert(host_entry_ids.end(), entries_by_query[query_id].begin(),
                              entries_by_query[query_id].end());
    }

    CudaBuffer<std::uint32_t> entry_offsets_buffer =
        CudaBuffer<std::uint32_t>::CopyFromHost(host_entry_offsets);
    CudaBuffer<std::uint32_t> entry_ids_buffer =
        CudaBuffer<std::uint32_t>::CopyFromHost(host_entry_ids);
    return MaterializeTopologyResults(
        RunTopologySearchKernelBatchDeviceImpl(resources, distance_oracle, entry_offsets_buffer,
                                               entry_ids_buffer, nullptr, 0, query_offset,
                                               num_queries,
                                               params, nullptr, nullptr, nullptr, nullptr, 0),
        params, out_profile);
}


DeviceTopologyBatchResult RunTopologySearchKernelBatchDeviceMicrobatched(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    const TopologyMicrobatchExecutionConfig& execution,
    TopologyMicrobatchExecutionProfile* out_execution_profile) {
    const char* context = "RunTopologySearchKernelBatchDeviceMicrobatched";
    DeviceTopologyBatchResult result;
    result.num_queries = entry_batch.num_queries;
    if (entry_batch.num_queries == 0) {
        return result;
    }
    if (entry_batch.offsets.size() != entry_batch.num_queries + 1U ||
        params.search_width == 0U || params.search_width > kMaxSearchWidth ||
        params.top_l == 0U || execution.microbatch_queries == 0U ||
        execution.context_depth == 0U || execution.io_blocks == 0U ||
        execution.io_threads == 0U || execution.io_threads > 256U ||
        execution.io_threads % kWarpSize != 0U) {
        throw std::runtime_error(BuildErrorMessage(
            context, "invalid entry batch, search, or execution dimensions."));
    }
    if (entry_batch.num_queries > distance_oracle.num_queries()) {
        throw std::runtime_error(BuildErrorMessage(
            context, "entry batch exceeds query distance tables."));
    }
    if (resources.num_nodes() >= static_cast<std::uint64_t>(kRawNodeIdMask)) {
        throw std::runtime_error(BuildErrorMessage(
            context, "packed node_id requires num_nodes < 2^31 - 1."));
    }

    const std::uint32_t degree = resources.degree();
    const std::uint64_t cached_node_count =
        std::min<std::uint64_t>(params.topology_cached_node_count,
                                resources.num_nodes());
    if (cached_node_count >= resources.num_nodes()) {
        throw std::runtime_error(BuildErrorMessage(
            context, "microbatch I/O path requires a partial topology cache."));
    }
    const bool exact_reuse_enabled = params.enable_exact_reuse;
    const void* topology_ssd = resources.topology_device_read_handle();
    const void* combined_ssd = resources.combined_node_device_read_handle();
    const void* io_handle = exact_reuse_enabled ? combined_ssd : topology_ssd;
    if (io_handle == nullptr) {
        throw std::runtime_error(BuildErrorMessage(
            context, exact_reuse_enabled
                         ? "combined-node SSD provider is missing."
                         : "topology SSD provider is missing."));
    }

    const std::size_t candidate_limit =
        std::max(params.top_l,
                 params.candidate_queue_size == 0U
                     ? params.top_l
                     : params.candidate_queue_size);
    const std::size_t candidate_capacity = NextPowerOfTwo(candidate_limit);
    const std::size_t frontier_slots = params.search_width * degree;
    const std::size_t frontier_capacity =
        NextPowerOfTwo(frontier_slots == 0U ? 1U : frontier_slots);
    const std::size_t unexpanded_words = (params.top_l + 31U) / 32U;
    const std::size_t traversal_shared_bytes =
        (candidate_capacity + frontier_capacity) * sizeof(DeviceCandidate) +
        frontier_slots * sizeof(std::uint32_t) +
        unexpanded_words * sizeof(std::uint32_t) + alignof(vectype);
    const std::size_t select_shared_bytes =
        candidate_capacity * sizeof(DeviceCandidate) +
        unexpanded_words * sizeof(std::uint32_t) + alignof(std::uint32_t);
    int device = 0;
    ThrowIfCudaError(cudaGetDevice(&device), "cudaGetDevice");
    int max_shared_bytes = 0;
    ThrowIfCudaError(
        cudaDeviceGetAttribute(&max_shared_bytes,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
        "cudaDeviceGetAttribute(max shared memory)");
    if (traversal_shared_bytes > static_cast<std::size_t>(max_shared_bytes) ||
        select_shared_bytes > static_cast<std::size_t>(max_shared_bytes)) {
        throw std::runtime_error(BuildErrorMessage(
            context, "microbatch shared-memory requirement exceeds device limit."));
    }
    ConfigureKernelDynamicSharedMemory(
        topology_microbatch_init_kernel<kNoFixedNumChunks>,
        traversal_shared_bytes);
    ConfigureKernelDynamicSharedMemory(
        topology_microbatch_init_kernel<kSpecializedSearchPqNumChunks>,
        traversal_shared_bytes);
    ConfigureKernelDynamicSharedMemory(
        topology_microbatch_select_kernel, select_shared_bytes);
    ConfigureKernelDynamicSharedMemory(
        topology_microbatch_resume_kernel<kNoFixedNumChunks>,
        traversal_shared_bytes);
    ConfigureKernelDynamicSharedMemory(
        topology_microbatch_resume_kernel<kSpecializedSearchPqNumChunks>,
        traversal_shared_bytes);

    result.candidate_capacity = candidate_capacity;
    result.candidate_buffer = CudaBuffer<DeviceCandidate>::Allocate(
        entry_batch.num_queries * candidate_capacity);
    result.stats_buffer = CudaBuffer<DeviceSearchStats>::Allocate(
        entry_batch.num_queries);
    result.profile_buffer = CudaBuffer<DeviceProfileCycles>::Allocate(
        entry_batch.num_queries);

    DeviceTraversalExactReuseConfig exact_reuse{};
    TopologyMicrobatchIoConfig io_config{};
    io_config.num_nodes = resources.num_nodes();
    io_config.degree = degree;
    io_config.queries = params.exact_reuse_device_queries;
    CudaBuffer<unsigned long long> validation_mismatches;
    if (const char* env =
            std::getenv("TOPOANNS_VALIDATE_MICROBATCH_TOPOLOGY")) {
        if (env[0] != '\0' && env[0] != '0') {
            validation_mismatches =
                CudaBuffer<unsigned long long>::Allocate(1U);
            ThrowIfCudaError(
                cudaMemset(validation_mismatches.get(), 0,
                           sizeof(unsigned long long)),
                "cudaMemset(microbatch validation mismatches)");
            io_config.validation_topology =
                resources.device_topology_data();
            io_config.validation_mismatch_neighbors =
                validation_mismatches.get();
        }
    }
    if (exact_reuse_enabled) {
        if (params.exact_reuse_device_queries == nullptr ||
            (params.exact_reuse_query_dim != 96U &&
             params.exact_reuse_query_dim != 128U &&
             params.exact_reuse_query_dim != 512U &&
             params.exact_reuse_query_dim != 768U)) {
            throw std::runtime_error(BuildErrorMessage(
                context, "invalid exact-reuse query buffers or dimension."));
        }
        const std::size_t vector_bytes =
            params.exact_reuse_query_dim * sizeof(float);
        const std::size_t minimum_node_bytes =
            vector_bytes +
            (static_cast<std::size_t>(degree) + 1U) * sizeof(std::uint32_t);
        if (params.combined_node_bytes < minimum_node_bytes ||
            params.combined_nodes_per_page == 0U ||
            static_cast<std::size_t>(params.combined_node_bytes) *
                    params.combined_nodes_per_page >
                kDefaultPageSizeBytes) {
            throw std::runtime_error(BuildErrorMessage(
                context, "invalid combined-node record layout."));
        }
        std::size_t cache_capacity = params.exact_reuse_cache_capacity;
        if (cache_capacity == 0U) {
            cache_capacity = NextPowerOfTwo(std::max<std::size_t>(
                2U, static_cast<std::size_t>(params.max_expansions) * 2U));
        }
        if ((cache_capacity & (cache_capacity - 1U)) != 0U ||
            cache_capacity > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error(BuildErrorMessage(
                context, "exact-reuse cache capacity must be a uint32 power of two."));
        }
        result.exact_reuse_cache_capacity = cache_capacity;
        result.exact_reuse_node_ids = CudaBuffer<std::uint32_t>::Allocate(
            entry_batch.num_queries * cache_capacity);
        result.exact_reuse_distances = CudaBuffer<float>::Allocate(
            entry_batch.num_queries * cache_capacity);
        ThrowIfCudaError(
            cudaMemset(result.exact_reuse_node_ids.get(), 0xff,
                       result.exact_reuse_node_ids.size() * sizeof(std::uint32_t)),
            "cudaMemset(microbatch exact reuse node IDs)");
        exact_reuse.enabled = 1U;
        exact_reuse.query_dim =
            static_cast<std::uint32_t>(params.exact_reuse_query_dim);
        exact_reuse.vector_bytes = static_cast<std::uint32_t>(vector_bytes);
        exact_reuse.node_bytes = params.combined_node_bytes;
        exact_reuse.nodes_per_page = params.combined_nodes_per_page;
        exact_reuse.cache_capacity =
            static_cast<std::uint32_t>(cache_capacity);
        exact_reuse.combined_ssd = combined_ssd;
        exact_reuse.queries = params.exact_reuse_device_queries;
        exact_reuse.cache_node_ids = result.exact_reuse_node_ids.get();
        exact_reuse.cache_distances = result.exact_reuse_distances.get();
        io_config.combined_node_bytes = params.combined_node_bytes;
        io_config.combined_nodes_per_page = params.combined_nodes_per_page;
        io_config.query_dim =
            static_cast<std::uint32_t>(params.exact_reuse_query_dim);
    } else {
        const std::size_t topology_node_bytes =
            static_cast<std::size_t>(degree) * sizeof(std::uint32_t);
        if (topology_node_bytes == 0U ||
            kDefaultPageSizeBytes % topology_node_bytes != 0U) {
            throw std::runtime_error(BuildErrorMessage(
                context, "topology records must divide a 4KB page."));
        }
        io_config.topology_nodes_per_page = static_cast<std::uint32_t>(
            kDefaultPageSizeBytes / topology_node_bytes);
    }

    std::uint32_t candidate_stop_divisor = 1U;
    if (const char* env = std::getenv("TOPOANNS_CANDIDATE_STOP_DIVISOR")) {
        const auto parsed =
            static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
        if (parsed > 1U) {
            candidate_stop_divisor = parsed;
        }
    }
    std::uint32_t candidate_stop_min_prefix = 32U;
    if (const char* env = std::getenv("TOPOANNS_CANDIDATE_STOP_MIN_PREFIX")) {
        const auto parsed =
            static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
        if (parsed != 0U) {
            candidate_stop_min_prefix = parsed;
        }
    }
    std::uint32_t candidate_stop_prefix =
        static_cast<std::uint32_t>(params.top_l);
    if (candidate_stop_divisor > 1U) {
        candidate_stop_prefix = std::max(
            candidate_stop_min_prefix,
            static_cast<std::uint32_t>(params.top_l / candidate_stop_divisor));
        candidate_stop_prefix = std::min(
            candidate_stop_prefix, static_cast<std::uint32_t>(params.top_l));
    }
    if (const char* env = std::getenv("TOPOANNS_CANDIDATE_STOP_PREFIX")) {
        const auto parsed =
            static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10));
        if (parsed != 0U) {
            candidate_stop_prefix = std::min(
                parsed, static_cast<std::uint32_t>(params.top_l));
        }
    }
    std::uint32_t candidate_stop_use_expanded = 0U;
    if (const char* env = std::getenv("TOPOANNS_CANDIDATE_STOP_USE_EXPANDED")) {
        const std::string value = TrimAscii(env);
        if (!(value.empty() || value == "0" || value == "false" ||
              value == "FALSE")) {
            candidate_stop_use_expanded = 1U;
        }
    }
    DeviceLearnedStopConfig learned_stop{};
    if (const char* env = std::getenv("TOPOANNS_LEARNED_STOP_MODEL")) {
        learned_stop = LoadLearnedStopConfigFromFile(
            env, static_cast<std::uint32_t>(params.top_l),
            static_cast<std::uint32_t>(params.top_k));
    }
    ThrowIfCudaError(
        cudaMemcpyToSymbol(kDeviceTraversalLearnedStopConfig, &learned_stop,
                           sizeof(learned_stop)),
        "cudaMemcpyToSymbol(kDeviceTraversalLearnedStopConfig)");

    const std::size_t query_stride =
        distance_oracle.pq_index().host().num_chunks * kNumPqCentroids;
    const float* query_tables =
        distance_oracle.query_tables().device_tables().get();
    const std::uint8_t* pq_codes =
        distance_oracle.pq_index().device_codes();
    const std::size_t num_chunks =
        distance_oracle.pq_index().host().num_chunks;
    bool disable_pq_num_chunks_specialization = false;
    if (const char* env =
            std::getenv("TOPOANNS_DISABLE_PQ_NUMCHUNKS_SPECIALIZATION")) {
        disable_pq_num_chunks_specialization = env[0] == '1';
    }
    const bool use_specialized_num_chunks =
        !disable_pq_num_chunks_specialization &&
        num_chunks == kSpecializedSearchPqNumChunks;
    constexpr std::uint32_t frontier_pq_warps = 1U;

    struct MicrobatchContext {
        enum class Phase { kIdle, kSelectPending, kRoundPending };

        CudaBuffer<std::uint32_t> visited_hash;
        CudaBuffer<DeviceMicrobatchQueryState> query_states;
        CudaBuffer<std::uint32_t> selected_nodes;
        CudaBuffer<std::uint32_t> response_neighbors;
        CudaBuffer<float> response_exact_distances;
        CudaBuffer<DeviceTopologyIoRequest> requests;
        CudaBuffer<std::uint32_t> request_count;
        CudaBuffer<std::uint32_t> active_count;
        cudaStream_t compute_stream = nullptr;
        cudaEvent_t select_ready = nullptr;
        cudaEvent_t io_begin = nullptr;
        cudaEvent_t io_end = nullptr;
        cudaEvent_t round_done = nullptr;
        std::uint32_t* host_counts = nullptr;
        std::size_t query_offset = 0;
        std::size_t num_queries = 0;
        std::size_t rounds = 0;
        Phase phase = Phase::kIdle;

        explicit MicrobatchContext(std::size_t max_queries,
                                   std::size_t search_width,
                                   std::size_t degree)
            : visited_hash(CudaBuffer<std::uint32_t>::Allocate(
                  max_queries * kVisitedHashWords)),
              query_states(CudaBuffer<DeviceMicrobatchQueryState>::Allocate(
                  max_queries)),
              selected_nodes(CudaBuffer<std::uint32_t>::Allocate(
                  max_queries * search_width)),
              response_neighbors(CudaBuffer<std::uint32_t>::Allocate(
                  max_queries * search_width * degree)),
              response_exact_distances(CudaBuffer<float>::Allocate(
                  max_queries * search_width)),
              requests(CudaBuffer<DeviceTopologyIoRequest>::Allocate(
                  max_queries * search_width)),
              request_count(CudaBuffer<std::uint32_t>::Allocate(1U)),
              active_count(CudaBuffer<std::uint32_t>::Allocate(1U)) {
            ThrowIfCudaError(
                cudaStreamCreateWithFlags(&compute_stream,
                                          cudaStreamNonBlocking),
                "cudaStreamCreate(microbatch compute)");
            ThrowIfCudaError(
                cudaEventCreateWithFlags(&select_ready, cudaEventDisableTiming),
                "cudaEventCreate(select ready)");
            ThrowIfCudaError(cudaEventCreate(&io_begin),
                             "cudaEventCreate(io begin)");
            ThrowIfCudaError(cudaEventCreate(&io_end),
                             "cudaEventCreate(io end)");
            ThrowIfCudaError(
                cudaEventCreateWithFlags(&round_done, cudaEventDisableTiming),
                "cudaEventCreate(round done)");
            ThrowIfCudaError(
                cudaHostAlloc(reinterpret_cast<void**>(&host_counts),
                              2U * sizeof(std::uint32_t),
                              cudaHostAllocPortable),
                "cudaHostAlloc(microbatch counts)");
        }

        ~MicrobatchContext() {
            if (compute_stream != nullptr) {
                cudaStreamDestroy(compute_stream);
            }
            if (select_ready != nullptr) {
                cudaEventDestroy(select_ready);
            }
            if (io_begin != nullptr) {
                cudaEventDestroy(io_begin);
            }
            if (io_end != nullptr) {
                cudaEventDestroy(io_end);
            }
            if (round_done != nullptr) {
                cudaEventDestroy(round_done);
            }
            if (host_counts != nullptr) {
                cudaFreeHost(host_counts);
            }
        }
    };

    const std::size_t slot_count = std::min(
        execution.context_depth,
        (entry_batch.num_queries + execution.microbatch_queries - 1U) /
            execution.microbatch_queries);
    std::vector<std::unique_ptr<MicrobatchContext>> slots;
    slots.reserve(slot_count);
    for (std::size_t slot = 0; slot < slot_count; ++slot) {
        slots.push_back(std::make_unique<MicrobatchContext>(
            execution.microbatch_queries, params.search_width, degree));
    }

    cudaStream_t io_stream = nullptr;
    int least_priority = 0;
    int greatest_priority = 0;
    ThrowIfCudaError(
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority),
        "cudaDeviceGetStreamPriorityRange");
    ThrowIfCudaError(
        cudaStreamCreateWithPriority(&io_stream, cudaStreamNonBlocking,
                                     greatest_priority),
        "cudaStreamCreateWithPriority(microbatch I/O)");

    TopologyMicrobatchExecutionProfile execution_profile{};
    execution_profile.min_nonempty_batch_requests =
        std::numeric_limits<std::size_t>::max();
    std::size_t next_query_offset = 0U;
    std::size_t active_slots = 0U;

    const auto launch_select = [&](MicrobatchContext& slot) {
        ThrowIfCudaError(
            cudaMemsetAsync(slot.request_count.get(), 0,
                            sizeof(std::uint32_t), slot.compute_stream),
            "cudaMemsetAsync(microbatch request count)");
        ThrowIfCudaError(
            cudaMemsetAsync(slot.active_count.get(), 0,
                            sizeof(std::uint32_t), slot.compute_stream),
            "cudaMemsetAsync(microbatch active count)");
        topology_microbatch_select_kernel
            <<<slot.num_queries, kTopologyBlockThreads, select_shared_bytes,
               slot.compute_stream>>>(
                resources.device_topology_data(), cached_node_count,
                resources.num_nodes(), degree, slot.query_offset,
                slot.num_queries, static_cast<std::uint32_t>(params.top_l),
                static_cast<std::uint32_t>(params.search_width),
                static_cast<std::uint32_t>(params.max_expansions),
                static_cast<std::uint32_t>(candidate_capacity),
                candidate_stop_prefix, candidate_stop_use_expanded,
                learned_stop, result.candidate_buffer.get(),
                result.stats_buffer.get(), result.profile_buffer.get(),
                slot.query_states.get(), slot.selected_nodes.get(),
                slot.response_neighbors.get(),
                slot.response_exact_distances.get(), slot.requests.get(),
                slot.request_count.get(), slot.requests.size(),
                slot.active_count.get());
        ThrowIfCudaError(cudaGetLastError(),
                         "topology_microbatch_select_kernel");
        ThrowIfCudaError(cudaEventRecord(slot.select_ready,
                                         slot.compute_stream),
                         "cudaEventRecord(select ready)");
        slot.phase = MicrobatchContext::Phase::kSelectPending;
    };

    const auto activate_slot = [&](MicrobatchContext& slot) {
        slot.query_offset = next_query_offset;
        slot.num_queries = std::min(
            execution.microbatch_queries,
            entry_batch.num_queries - next_query_offset);
        next_query_offset += slot.num_queries;
        slot.rounds = 0U;
        const auto launch_init = [&](auto fixed_chunks) {
            constexpr std::size_t kFixedChunks =
                decltype(fixed_chunks)::value;
            topology_microbatch_init_kernel<kFixedChunks>
                <<<slot.num_queries, kTopologyBlockThreads,
                   traversal_shared_bytes, slot.compute_stream>>>(
                    resources.num_nodes(), pq_codes, num_chunks, query_tables,
                    static_cast<std::uint32_t>(query_stride),
                    entry_batch.offsets.get(), entry_batch.ids.get(),
                    slot.query_offset, slot.num_queries,
                    static_cast<std::uint32_t>(params.top_l),
                    static_cast<std::uint32_t>(candidate_capacity),
                    static_cast<std::uint32_t>(frontier_slots),
                    static_cast<std::uint32_t>(frontier_capacity),
                    frontier_pq_warps, learned_stop, slot.visited_hash.get(),
                    result.candidate_buffer.get(), result.stats_buffer.get(),
                    result.profile_buffer.get(), slot.query_states.get());
        };
        if (use_specialized_num_chunks) {
            launch_init(std::integral_constant<
                        std::size_t, kSpecializedSearchPqNumChunks>{});
        } else {
            launch_init(std::integral_constant<
                        std::size_t, kNoFixedNumChunks>{});
        }
        ThrowIfCudaError(cudaGetLastError(),
                         "topology_microbatch_init_kernel");
        launch_select(slot);
    };

    const auto begin = std::chrono::steady_clock::now();
    for (auto& slot : slots) {
        activate_slot(*slot);
        ++active_slots;
    }

    try {
        while (active_slots != 0U) {
            bool progressed = false;
            for (auto& slot_ptr : slots) {
                MicrobatchContext& slot = *slot_ptr;
                if (slot.phase == MicrobatchContext::Phase::kIdle) {
                    continue;
                }
                if (slot.phase ==
                    MicrobatchContext::Phase::kSelectPending) {
                    const cudaError_t status =
                        cudaEventQuery(slot.select_ready);
                    if (status == cudaErrorNotReady) {
                        continue;
                    }
                    ThrowIfCudaError(status, "cudaEventQuery(select ready)");
                    ThrowIfCudaError(cudaEventRecord(slot.io_begin, io_stream),
                                     "cudaEventRecord(microbatch I/O begin)");
                    LaunchBamTopologyMicrobatchReads(
                        io_handle, slot.requests.get(),
                        slot.request_count.get(), slot.requests.size(),
                        io_config, slot.response_neighbors.get(),
                        slot.response_exact_distances.get(),
                        execution.io_blocks, execution.io_threads, io_stream);
                    ThrowIfCudaError(cudaEventRecord(slot.io_end, io_stream),
                                     "cudaEventRecord(microbatch I/O end)");
                    ThrowIfCudaError(
                        cudaStreamWaitEvent(slot.compute_stream, slot.io_end, 0),
                        "cudaStreamWaitEvent(microbatch I/O end)");
                    const auto launch_resume = [&](auto fixed_chunks) {
                        constexpr std::size_t kFixedChunks =
                            decltype(fixed_chunks)::value;
                        topology_microbatch_resume_kernel<kFixedChunks>
                            <<<slot.num_queries, kTopologyBlockThreads,
                               traversal_shared_bytes, slot.compute_stream>>>(
                                resources.num_nodes(), degree, pq_codes,
                                num_chunks, query_tables,
                                static_cast<std::uint32_t>(query_stride),
                                slot.query_offset, slot.num_queries,
                                static_cast<std::uint32_t>(params.top_l),
                                static_cast<std::uint32_t>(params.search_width),
                                static_cast<std::uint32_t>(candidate_capacity),
                                static_cast<std::uint32_t>(frontier_slots),
                                static_cast<std::uint32_t>(frontier_capacity),
                                frontier_pq_warps, learned_stop, exact_reuse,
                                slot.visited_hash.get(),
                                result.candidate_buffer.get(),
                                result.stats_buffer.get(),
                                result.profile_buffer.get(),
                                slot.query_states.get(),
                                slot.selected_nodes.get(),
                                slot.response_neighbors.get(),
                                slot.response_exact_distances.get());
                    };
                    if (use_specialized_num_chunks) {
                        launch_resume(std::integral_constant<
                                      std::size_t,
                                      kSpecializedSearchPqNumChunks>{});
                    } else {
                        launch_resume(std::integral_constant<
                                      std::size_t, kNoFixedNumChunks>{});
                    }
                    ThrowIfCudaError(cudaGetLastError(),
                                     "topology_microbatch_resume_kernel");
                    ThrowIfCudaError(
                        cudaMemcpyAsync(slot.host_counts,
                                        slot.request_count.get(),
                                        sizeof(std::uint32_t),
                                        cudaMemcpyDeviceToHost,
                                        slot.compute_stream),
                        "cudaMemcpyAsync(microbatch request count)");
                    ThrowIfCudaError(
                        cudaMemcpyAsync(slot.host_counts + 1U,
                                        slot.active_count.get(),
                                        sizeof(std::uint32_t),
                                        cudaMemcpyDeviceToHost,
                                        slot.compute_stream),
                        "cudaMemcpyAsync(microbatch active count)");
                    ThrowIfCudaError(cudaEventRecord(slot.round_done,
                                                     slot.compute_stream),
                                     "cudaEventRecord(microbatch round done)");
                    slot.phase =
                        MicrobatchContext::Phase::kRoundPending;
                    progressed = true;
                    continue;
                }

                const cudaError_t status = cudaEventQuery(slot.round_done);
                if (status == cudaErrorNotReady) {
                    continue;
                }
                ThrowIfCudaError(status, "cudaEventQuery(round done)");
                float io_ms = 0.0f;
                ThrowIfCudaError(
                    cudaEventElapsedTime(&io_ms, slot.io_begin, slot.io_end),
                    "cudaEventElapsedTime(microbatch I/O)");
                const std::size_t requests = slot.host_counts[0];
                const std::size_t active_queries = slot.host_counts[1];
                if (requests > slot.requests.size()) {
                    throw std::runtime_error(BuildErrorMessage(
                        context, "microbatch request buffer overflow."));
                }
                execution_profile.summed_io_kernel_ms += io_ms;
                execution_profile.logical_io_requests += requests;
                ++execution_profile.io_batches;
                if (requests != 0U) {
                    ++execution_profile.nonempty_io_batches;
                    execution_profile.min_nonempty_batch_requests = std::min(
                        execution_profile.min_nonempty_batch_requests,
                        requests);
                    execution_profile.max_batch_requests = std::max(
                        execution_profile.max_batch_requests, requests);
                }
                ++slot.rounds;
                if (slot.rounds > params.max_expansions + 2U) {
                    throw std::runtime_error(BuildErrorMessage(
                        context, "microbatch traversal exceeded round guard."));
                }
                if (active_queries == 0U) {
                    if (next_query_offset < entry_batch.num_queries) {
                        activate_slot(slot);
                    } else {
                        slot.phase = MicrobatchContext::Phase::kIdle;
                        --active_slots;
                    }
                } else {
                    launch_select(slot);
                }
                progressed = true;
            }
            if (!progressed) {
                std::this_thread::yield();
            }
        }
        ThrowIfCudaError(cudaStreamSynchronize(io_stream),
                         "cudaStreamSynchronize(microbatch I/O)");
        ThrowIfCudaError(cudaDeviceSynchronize(),
                         "cudaDeviceSynchronize(microbatch topology)");
    } catch (...) {
        cudaDeviceSynchronize();
        cudaStreamDestroy(io_stream);
        throw;
    }
    const auto end = std::chrono::steady_clock::now();
    cudaStreamDestroy(io_stream);

    execution_profile.wall_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    if (execution_profile.nonempty_io_batches == 0U) {
        execution_profile.min_nonempty_batch_requests = 0U;
    }
    if (!validation_mismatches.empty()) {
        execution_profile.validation_mismatch_neighbors =
            validation_mismatches.CopyToHost().front();
    }
    result.kernel_ms = execution_profile.wall_ms;
    if (out_execution_profile != nullptr) {
        *out_execution_profile = execution_profile;
    }
    return result;
}
DeviceTopologyBatchResult RunTopologySearchKernelBatchDevice(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    cudaStream_t stream) {
    if (entry_batch.offsets.size() != entry_batch.num_queries + 1) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatchDevice", "device entry offsets size is invalid."));
    }
    return RunTopologySearchKernelBatchDeviceImpl(resources, distance_oracle, entry_batch.offsets,
                                                  entry_batch.ids, nullptr, 0, 0,
                                                  entry_batch.num_queries, params, nullptr,
                                                  nullptr, nullptr, nullptr, stream);
}

DeviceTopologyBatchResult RunTopologySearchKernelBatchDeviceWithLaunchCallback(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    TopologyLaunchHostCallback launch_callback,
    void* launch_callback_context,
    cudaStream_t stream) {
    if (entry_batch.offsets.size() != entry_batch.num_queries + 1) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatchDevice", "device entry offsets size is invalid."));
    }
    return RunTopologySearchKernelBatchDeviceImpl(resources, distance_oracle, entry_batch.offsets,
                                                  entry_batch.ids, nullptr, 0, 0,
                                                  entry_batch.num_queries, params, nullptr,
                                                  nullptr, launch_callback, launch_callback_context, stream);
}

DeviceTopologyBatchResult RunTopologySearchKernelBatchDevice(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params) {
    if (entry_batch.offsets.size() != entry_batch.num_queries + 1) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatchDevice", "device entry offsets size is invalid."));
    }
    return RunTopologySearchKernelBatchDeviceImpl(resources, distance_oracle, entry_batch.offsets,
                                                  entry_batch.ids, nullptr, 0, 0,
                                                  entry_batch.num_queries,
                                                  params, nullptr, nullptr, nullptr, nullptr, 0);
}

DeviceTopologyBatchResult RunTopologySearchKernelBatchDevice(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const CudaBuffer<std::uint32_t>& gt_ids,
    std::uint32_t gt_topk,
    const TopologySearchParams& params) {
    if (entry_batch.offsets.size() != entry_batch.num_queries + 1) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatchDevice", "device entry offsets size is invalid."));
    }
    return RunTopologySearchKernelBatchDeviceImpl(resources, distance_oracle, entry_batch.offsets,
                                                  entry_batch.ids, &gt_ids, gt_topk, 0,
                                                  entry_batch.num_queries, params, nullptr,
                                                  nullptr, nullptr, nullptr, 0);
}

DeviceTopologyBatchResult RunTopologySearchKernelBatchDeviceDebug(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    const DeviceTopologyDebugConfig& debug_config,
    DeviceTopologyDebugTrace* out_debug_trace) {
    if (entry_batch.offsets.size() != entry_batch.num_queries + 1) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatchDeviceDebug", "device entry offsets size is invalid."));
    }
    if (out_debug_trace == nullptr) {
        throw std::runtime_error(BuildErrorMessage(
            "RunTopologySearchKernelBatchDeviceDebug", "out_debug_trace must not be null."));
    }
    return RunTopologySearchKernelBatchDeviceImpl(resources, distance_oracle, entry_batch.offsets,
                                                  entry_batch.ids, nullptr, 0, 0,
                                                  entry_batch.num_queries, params, &debug_config,
                                                  out_debug_trace, nullptr, nullptr, 0);
}

std::vector<TopologySearchResult> RunTopologySearchKernelBatch(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    TopologySearchBatchProfile* out_profile) {
    return MaterializeTopologyResults(
        RunTopologySearchKernelBatchDevice(resources, distance_oracle, entry_batch, params), params,
        out_profile);
}

}  // namespace topoanns::detail
