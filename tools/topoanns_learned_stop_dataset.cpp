#include "topoanns/bam_runtime_config.hpp"
#include "topoanns/bam_vector_provider.hpp"
#include "topoanns/bvecs_io.hpp"
#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/rvq_entry_provider.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/topoanns_search.hpp"
#include "topoanns/vector_store_builder.hpp"
#include "../src/search/fused_rerank_device.hpp"
#include "../src/search/topology_search_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::filesystem::path index_dir;
    std::filesystem::path rvq_model;
    std::filesystem::path learn_bvecs;
    std::filesystem::path query_bin;
    std::filesystem::path gt_bin;
    std::filesystem::path output_csv;
    std::filesystem::path pq2_pivots;
    std::filesystem::path pq2_codes;
    std::filesystem::path pq2_error_bounds;
    std::size_t query_start = 0;
    std::size_t num_queries = 0;
    std::size_t batch_size = 256;
    std::size_t top_k = 10;
    std::vector<std::size_t> top_l_values;
    std::vector<std::size_t> rerank_top_n_values;
    std::size_t search_width = 2;
    std::size_t max_expansions = 4096;
    std::size_t rvq_entry_count = 128;
    std::optional<std::filesystem::path> bam_config_path;
    topoanns::BamRuntimeConfigOverrides bam_overrides;
    bool allow_bam_controller_override = false;
    std::size_t bam_device_offset_bytes = 0;
    bool use_pq2_refine = false;
    bool use_pq2_bound_filter = false;
    bool use_early_stop = false;
    bool use_learned_stop = false;
    std::size_t rerank_early_stop_min_prefix = 32;
    std::size_t rerank_early_stop_patience_tiles = 1;
    std::size_t rerank_rank_tile_size = 10;
    std::filesystem::path rerank_learned_stop_model;
};

struct QuerySnapshot {
    std::vector<std::uint32_t> label_topk_ids;
    std::vector<std::uint32_t> feature_topk_ids;
    std::size_t expanded_nodes = 0;
    std::size_t visited_nodes = 0;
    std::size_t valid_candidates = 0;
    std::size_t best_unexpanded_rank = 0;
    std::size_t unexpanded_next64 = 0;
    std::size_t unexpanded_next256 = 0;
    float d1 = 0.0f;
    float dk = 0.0f;
    float boundary = 0.0f;
    float gap32 = 0.0f;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_learned_stop_dataset"
        << " --index-dir <path>"
        << " --rvq-model <path>"
        << " (--learn-bvecs <path> | --query-bin <path> --gt-bin <path>)"
        << " --output-csv <path>"
        << " --num-queries <count>"
        << " --top-l-values <csv>"
        << " --rerank-top-n-values <csv>"
        << " [--query-start <count>]"
        << " [--batch-size <count>]"
        << " [--top-k <count>]"
        << " [--search-width <count>]"
        << " [--max-expansions <count>]"
        << " [--rvq-entry-count <count>]"
        << " [--pq2-pivots <path> --pq2-codes <path> [--pq2-error-bounds <path>] --rerank-use-pq2]"
        << " [--rerank-use-pq2-bound-filter]"
        << " [--rerank-use-early-stop]"
        << " [--rerank-use-learned-stop --rerank-learned-stop-model <path>]"
        << " [--rerank-early-stop-min-prefix <count>]"
        << " [--rerank-early-stop-patience-tiles <count>]"
        << " [--rerank-rank-tile-size <count>]"
        << " [--bam-config-path <path>]"
        << " [--allow-bam-controller-override]"
        << " [--bam-controller-path <path>]"
        << std::endl;
    std::exit(1);
}

std::vector<std::size_t> ParseCsvSizes(const std::string& text) {
    std::vector<std::size_t> values;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token =
            text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            values.push_back(std::stoull(token));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return values;
}

void RequireL40Device(std::uint32_t device_id, const char* context) {
    cudaDeviceProp props{};
    const cudaError_t status = cudaGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": cudaGetDeviceProperties failed.");
    }
    const std::string name(props.name);
    if (name.find("L40") == std::string::npos) {
        throw std::runtime_error(std::string(context) + ": expected L40 but got " + name);
    }
}

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        auto read_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (flag == "--index-dir") {
            args.index_dir = read_value("--index-dir");
        } else if (flag == "--rvq-model") {
            args.rvq_model = read_value("--rvq-model");
        } else if (flag == "--learn-bvecs") {
            args.learn_bvecs = read_value("--learn-bvecs");
        } else if (flag == "--query-bin") {
            args.query_bin = read_value("--query-bin");
        } else if (flag == "--gt-bin") {
            args.gt_bin = read_value("--gt-bin");
        } else if (flag == "--output-csv") {
            args.output_csv = read_value("--output-csv");
        } else if (flag == "--pq2-pivots") {
            args.pq2_pivots = read_value("--pq2-pivots");
        } else if (flag == "--pq2-codes") {
            args.pq2_codes = read_value("--pq2-codes");
        } else if (flag == "--pq2-error-bounds") {
            args.pq2_error_bounds = read_value("--pq2-error-bounds");
        } else if (flag == "--query-start") {
            args.query_start = std::stoull(read_value("--query-start"));
        } else if (flag == "--num-queries") {
            args.num_queries = std::stoull(read_value("--num-queries"));
        } else if (flag == "--batch-size") {
            args.batch_size = std::stoull(read_value("--batch-size"));
        } else if (flag == "--top-k") {
            args.top_k = std::stoull(read_value("--top-k"));
        } else if (flag == "--top-l-values") {
            args.top_l_values = ParseCsvSizes(read_value("--top-l-values"));
        } else if (flag == "--rerank-top-n-values") {
            args.rerank_top_n_values = ParseCsvSizes(read_value("--rerank-top-n-values"));
        } else if (flag == "--search-width") {
            args.search_width = std::stoull(read_value("--search-width"));
        } else if (flag == "--max-expansions") {
            args.max_expansions = std::stoull(read_value("--max-expansions"));
        } else if (flag == "--rvq-entry-count") {
            args.rvq_entry_count = std::stoull(read_value("--rvq-entry-count"));
        } else if (flag == "--bam-config-path") {
            args.bam_config_path = read_value("--bam-config-path");
        } else if (flag == "--allow-bam-controller-override") {
            args.allow_bam_controller_override = true;
        } else if (flag == "--bam-controller-path") {
            args.bam_overrides.controller_path = read_value("--bam-controller-path");
        } else if (flag == "--bam-device-offset-bytes") {
            args.bam_device_offset_bytes = std::stoull(read_value("--bam-device-offset-bytes"));
        } else if (flag == "--bam-page-cache-bytes") {
            args.bam_overrides.page_cache_size_bytes =
                std::stoull(read_value("--bam-page-cache-bytes"));
        } else if (flag == "--bam-queue-depth") {
            args.bam_overrides.queue_depth =
                std::stoull(read_value("--bam-queue-depth"));
        } else if (flag == "--bam-num-queues") {
            args.bam_overrides.num_queues =
                std::stoull(read_value("--bam-num-queues"));
        } else if (flag == "--bam-cuda-device") {
            args.bam_overrides.cuda_device = static_cast<std::uint32_t>(
                std::stoul(read_value("--bam-cuda-device")));
        } else if (flag == "--rerank-use-pq2") {
            args.use_pq2_refine = true;
        } else if (flag == "--rerank-use-pq2-bound-filter") {
            args.use_pq2_bound_filter = true;
        } else if (flag == "--rerank-use-early-stop") {
            args.use_early_stop = true;
        } else if (flag == "--rerank-use-learned-stop") {
            args.use_learned_stop = true;
        } else if (flag == "--rerank-learned-stop-model") {
            args.rerank_learned_stop_model = read_value("--rerank-learned-stop-model");
        } else if (flag == "--rerank-early-stop-min-prefix") {
            args.rerank_early_stop_min_prefix =
                std::stoull(read_value("--rerank-early-stop-min-prefix"));
        } else if (flag == "--rerank-early-stop-patience-tiles") {
            args.rerank_early_stop_patience_tiles =
                std::stoull(read_value("--rerank-early-stop-patience-tiles"));
        } else if (flag == "--rerank-rank-tile-size") {
            args.rerank_rank_tile_size = std::stoull(read_value("--rerank-rank-tile-size"));
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }

    const bool has_learn_bvecs = !args.learn_bvecs.empty();
    const bool has_query_gt = !args.query_bin.empty() && !args.gt_bin.empty();
    if (args.index_dir.empty() || args.rvq_model.empty() || args.output_csv.empty() ||
        args.num_queries == 0 || args.top_l_values.empty() ||
        args.rerank_top_n_values.empty()) {
        Usage();
    }
    if (!has_learn_bvecs && !has_query_gt) {
        Usage();
    }
    if (!args.query_bin.empty() != !args.gt_bin.empty()) {
        Usage();
    }
    if (args.top_l_values.size() != args.rerank_top_n_values.size()) {
        throw std::runtime_error("top-l-values and rerank-top-n-values must match.");
    }
    if (args.pq2_pivots.empty() != args.pq2_codes.empty()) {
        throw std::runtime_error("pq2-pivots and pq2-codes must be provided together.");
    }
    if (args.use_pq2_refine && args.pq2_pivots.empty()) {
        throw std::runtime_error("rerank-use-pq2 requires pq2-pivots and pq2-codes.");
    }
    if (args.use_pq2_bound_filter && args.pq2_error_bounds.empty()) {
        throw std::runtime_error(
            "rerank-use-pq2-bound-filter requires pq2-error-bounds.");
    }
    if (args.use_learned_stop && args.rerank_learned_stop_model.empty()) {
        throw std::runtime_error(
            "rerank-use-learned-stop requires rerank-learned-stop-model.");
    }
    if (args.use_learned_stop && args.use_early_stop) {
        throw std::runtime_error(
            "rerank-use-learned-stop cannot be combined with rerank-use-early-stop.");
    }
    return args;
}

struct FloatMatrix {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<float> values;
};

struct IntMatrix {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<std::int32_t> values;
};

class HostVectorStoreReader {
public:
    HostVectorStoreReader(const std::filesystem::path& path,
                          const topoanns::VectorPageLayout& layout)
        : layout_(layout), file_(path, std::ios::binary) {
        if (!file_.is_open()) {
            throw std::runtime_error("Failed to open vector store: " + path.string());
        }
        scratch_.resize(layout_.vector_bytes());
    }

    const float* ReadFloat32(std::uint32_t node_id) {
        const topoanns::VectorPageAddress address = layout_.Resolve(node_id);
        file_.seekg(static_cast<std::streamoff>(address.byte_offset), std::ios::beg);
        file_.read(reinterpret_cast<char*>(scratch_.data()),
                   static_cast<std::streamsize>(scratch_.size()));
        if (!file_) {
            throw std::runtime_error("Failed to read vector store payload.");
        }
        return reinterpret_cast<const float*>(scratch_.data());
    }

private:
    topoanns::VectorPageLayout layout_;
    std::ifstream file_;
    std::vector<std::uint8_t> scratch_;
};

float ComputeSquaredL2(const float* lhs, const float* rhs, std::size_t dim) {
    float distance = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float delta = lhs[i] - rhs[i];
        distance += delta * delta;
    }
    return distance;
}

topoanns::RerankQueryResult BuildCpuExactRerankResult(
    const topoanns::detail::DeviceTopologyCandidate* candidates,
    std::size_t valid_candidates,
    std::size_t top_n,
    const float* query,
    std::size_t dim,
    std::size_t top_k,
    HostVectorStoreReader& vector_reader) {
    topoanns::RerankQueryResult result;
    const std::size_t exact_count = std::min(valid_candidates, top_n);
    result.sorted_candidates.reserve(exact_count);
    for (std::size_t i = 0; i < exact_count; ++i) {
        const auto& candidate = candidates[i];
        if (!candidate.valid()) {
            continue;
        }
        const std::uint32_t raw_node_id = candidate.raw_node_id();
        const float* vector = vector_reader.ReadFloat32(raw_node_id);
        topoanns::RankedCandidate exact;
        exact.node_id = raw_node_id;
        exact.distance = ComputeSquaredL2(query, vector, dim);
        exact.expanded = candidate.expanded();
        result.sorted_candidates.push_back(exact);
    }
    std::sort(result.sorted_candidates.begin(), result.sorted_candidates.end(),
              [](const topoanns::RankedCandidate& lhs, const topoanns::RankedCandidate& rhs) {
                  if (lhs.distance != rhs.distance) {
                      return lhs.distance < rhs.distance;
                  }
                  return lhs.node_id < rhs.node_id;
              });
    const std::size_t topk_count = std::min(top_k, result.sorted_candidates.size());
    result.topk.assign(result.sorted_candidates.begin(),
                       result.sorted_candidates.begin() + topk_count);
    return result;
}

FloatMatrix LoadFloatMatrix(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open float matrix: " + path.string());
    }
    FloatMatrix matrix;
    in.read(reinterpret_cast<char*>(&matrix.rows), sizeof(matrix.rows));
    in.read(reinterpret_cast<char*>(&matrix.cols), sizeof(matrix.cols));
    if (!in.good() || matrix.rows == 0 || matrix.cols == 0) {
        throw std::runtime_error("Invalid float matrix header: " + path.string());
    }
    matrix.values.resize(static_cast<std::size_t>(matrix.rows) * matrix.cols);
    in.read(reinterpret_cast<char*>(matrix.values.data()),
            static_cast<std::streamsize>(matrix.values.size() * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(matrix.values.size() * sizeof(float))) {
        throw std::runtime_error("Short read in float matrix: " + path.string());
    }
    return matrix;
}

IntMatrix LoadIntMatrix(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open int matrix: " + path.string());
    }
    IntMatrix matrix;
    in.read(reinterpret_cast<char*>(&matrix.rows), sizeof(matrix.rows));
    in.read(reinterpret_cast<char*>(&matrix.cols), sizeof(matrix.cols));
    if (!in.good() || matrix.rows == 0 || matrix.cols == 0) {
        throw std::runtime_error("Invalid int matrix header: " + path.string());
    }

    const std::uint64_t ids_bytes =
        static_cast<std::uint64_t>(matrix.rows) * matrix.cols * sizeof(std::int32_t);
    const std::uint64_t plain_xbin_bytes = 8ULL + ids_bytes;
    const std::uint64_t diskann_truthset_bytes =
        8ULL + ids_bytes +
        static_cast<std::uint64_t>(matrix.rows) * matrix.cols * sizeof(float);
    const std::uint64_t file_bytes = std::filesystem::file_size(path);

    matrix.values.resize(static_cast<std::size_t>(matrix.rows) * matrix.cols);
    in.read(reinterpret_cast<char*>(matrix.values.data()),
            static_cast<std::streamsize>(ids_bytes));
    if (in.gcount() != static_cast<std::streamsize>(ids_bytes)) {
        throw std::runtime_error("Short read in int matrix ids: " + path.string());
    }
    if (file_bytes != plain_xbin_bytes && file_bytes != diskann_truthset_bytes) {
        throw std::runtime_error("Unsupported GT file layout: " + path.string());
    }
    return matrix;
}

std::vector<std::uint32_t> BuildSparsePrefixLadder(std::size_t top_l) {
    const std::array<std::size_t, 8> anchors = {16, 32, 64, 128, 256, 512, 1024, 2048};
    const std::array<std::size_t, 7> inserted = {0, 1, 1, 1, 2, 4, 8};
    std::vector<std::uint32_t> ladder;
    for (std::size_t interval = 0; interval < inserted.size(); ++interval) {
        const std::size_t left = anchors[interval];
        const std::size_t right = anchors[interval + 1];
        if (left >= top_l) {
            break;
        }
        ladder.push_back(static_cast<std::uint32_t>(left));
        for (std::size_t i = 1; i <= inserted[interval]; ++i) {
            const double ratio =
                static_cast<double>(i) / static_cast<double>(inserted[interval] + 1);
            std::size_t prefix = static_cast<std::size_t>(
                std::llround(static_cast<double>(left) +
                             (static_cast<double>(right - left) * ratio)));
            prefix = ((prefix + 7U) / 16U) * 16U;
            if (prefix > left && prefix < right && prefix < top_l) {
                ladder.push_back(static_cast<std::uint32_t>(prefix));
            }
        }
    }
    std::sort(ladder.begin(), ladder.end());
    ladder.erase(std::unique(ladder.begin(), ladder.end()), ladder.end());
    return ladder;
}

bool TopKEquals(const QuerySnapshot& lhs, const QuerySnapshot& rhs) {
    return lhs.label_topk_ids == rhs.label_topk_ids;
}

float RecallAtK(const QuerySnapshot& snapshot,
                const std::int32_t* gt_row,
                std::size_t top_k) {
    if (top_k == 0) {
        return 0.0f;
    }
    std::size_t matched = 0;
    for (const std::uint32_t node_id : snapshot.label_topk_ids) {
        for (std::size_t rank = 0; rank < top_k; ++rank) {
            if (static_cast<std::int32_t>(node_id) == gt_row[rank]) {
                ++matched;
                break;
            }
        }
    }
    return static_cast<float>(matched) / static_cast<float>(top_k);
}

float ComputeTopKChurn(const std::vector<std::uint32_t>& current_topk_ids,
                      const std::uint32_t* previous_topk_ids,
                      std::size_t top_k,
                      bool has_previous) {
    if (!has_previous) {
        return 1.0f;
    }
    std::size_t changed = 0;
    for (std::size_t i = 0; i < top_k; ++i) {
        const std::uint32_t current_id =
            i < current_topk_ids.size() ? current_topk_ids[i] : topoanns::kInvalidNodeId;
        bool found = false;
        for (std::size_t j = 0; j < top_k; ++j) {
            if (previous_topk_ids[j] == current_id) {
                found = true;
                break;
            }
        }
        changed += found ? 0U : 1U;
    }
    return static_cast<float>(changed) / static_cast<float>(top_k);
}

QuerySnapshot BuildSnapshot(
                           const topoanns::detail::DeviceTopologyCandidate* candidates,
                           std::size_t candidate_capacity,
                           const topoanns::detail::DeviceTopologySearchStats& topology_stats,
                           const topoanns::RerankQueryResult& rerank,
                           std::size_t top_k,
                           std::size_t top_l,
                           std::size_t prefix) {
    QuerySnapshot snapshot;
    snapshot.expanded_nodes = topology_stats.expanded_nodes;
    snapshot.visited_nodes = topology_stats.visited_nodes;
    snapshot.valid_candidates = topology_stats.valid_candidates;
    snapshot.label_topk_ids.reserve(rerank.topk.size());
    for (const auto& candidate : rerank.topk) {
        snapshot.label_topk_ids.push_back(candidate.node_id);
    }
    if (candidate_capacity == 0 || snapshot.valid_candidates < top_k ||
        snapshot.valid_candidates < prefix) {
        throw std::runtime_error("Topology snapshot is smaller than requested feature prefix.");
    }
    snapshot.feature_topk_ids.reserve(top_k);
    for (std::size_t i = 0; i < top_k; ++i) {
        snapshot.feature_topk_ids.push_back(candidates[i].raw_node_id());
    }
    snapshot.d1 = candidates[0].distance;
    snapshot.dk = candidates[top_k - 1].distance;
    snapshot.boundary = candidates[prefix - 1].distance;
    const std::size_t valid_prefix = std::min<std::size_t>(snapshot.valid_candidates, top_l);
    snapshot.best_unexpanded_rank = top_l;
    for (std::size_t i = 0; i < valid_prefix; ++i) {
        if (!candidates[i].expanded()) {
            snapshot.best_unexpanded_rank = i;
            break;
        }
    }
    const std::size_t next64_end = std::min(prefix + 64U, top_l);
    const std::size_t next256_end = std::min(prefix + 256U, top_l);
    for (std::size_t i = prefix; i < next64_end; ++i) {
        if (i < snapshot.valid_candidates && !candidates[i].expanded()) {
            ++snapshot.unexpanded_next64;
        }
    }
    for (std::size_t i = prefix; i < next256_end; ++i) {
        if (i < snapshot.valid_candidates && !candidates[i].expanded()) {
            ++snapshot.unexpanded_next256;
        }
    }
    const std::size_t gap32_index = std::min<std::size_t>(
        std::max<std::size_t>(top_k - 1U, 31U), prefix - 1U);
    snapshot.gap32 = candidates[gap32_index].distance - snapshot.dk;
    return snapshot;
}

std::vector<QuerySnapshot> RunConfiguration(const topoanns::SearchResources& resources,
                                            HostVectorStoreReader& vector_reader,
                                            topoanns::RvqModel& rvq_model,
                                            const std::vector<float>& queries,
                                            std::size_t num_queries,
                                            std::size_t batch_size,
                                            std::size_t rvq_entry_count,
                                            const topoanns::TopoAnnsSearchParams& params,
                                            std::size_t prefix_override) {
    if (prefix_override != 0U) {
        setenv("TOPOANNS_CANDIDATE_STOP_PREFIX",
               std::to_string(prefix_override).c_str(), 1);
        setenv("TOPOANNS_CANDIDATE_STOP_USE_EXPANDED", "1", 1);
    } else {
        unsetenv("TOPOANNS_CANDIDATE_STOP_PREFIX");
        unsetenv("TOPOANNS_CANDIDATE_STOP_USE_EXPANDED");
    }

    const std::size_t dim = resources.vector_store_header().dim;
    std::vector<QuerySnapshot> snapshots(num_queries);
    for (std::size_t query_offset = 0; query_offset < num_queries; query_offset += batch_size) {
        const std::size_t batch_queries =
            std::min(batch_size, num_queries - query_offset);
        const float* begin = queries.data() + query_offset * dim;
        const float* end = begin + batch_queries * dim;
        std::vector<float> batch_query_buffer(begin, end);
        topoanns::RvqEntryProfile entry_profile;
        topoanns::DeviceEntryBatch entry_batch = rvq_model.ComputeFloat32DeviceEntryBatch(
            batch_query_buffer, batch_queries, rvq_entry_count, 0, &entry_profile);
        (void)entry_profile;
        const topoanns::PqDistanceOracle oracle =
            topoanns::PqDistanceOracle::FromFloatQueries(resources, batch_query_buffer,
                                                         batch_queries);
        topoanns::TopologySearchParams topology_params = params.topology;
        topology_params.candidate_queue_size =
            std::max({topology_params.top_l,
                      topology_params.candidate_queue_size,
                      params.rerank.top_n});
        topoanns::detail::DeviceTopologyBatchResult topology_device =
            topoanns::detail::RunTopologySearchKernelBatchDevice(resources, oracle, entry_batch,
                                                                 topology_params);
        const std::vector<topoanns::detail::DeviceTopologySearchStats> host_stats =
            topology_device.stats_buffer.CopyToHost();
        const std::vector<topoanns::detail::DeviceTopologyCandidate> host_candidates =
            topology_device.candidate_buffer.CopyToHost();

        std::optional<topoanns::RerankBatchResult> device_rerank_result;
        if (resources.vector_page_provider().SupportsDeviceReads()) {
            device_rerank_result = topoanns::detail::RunBatchFloat32FromDeviceTopology(
                resources, topology_device, batch_query_buffer, batch_queries, params.rerank,
                nullptr);
        }

        std::vector<topoanns::detail::DeviceTopologyCandidate> rerank_candidates =
            host_candidates;
        std::size_t rerank_candidate_capacity = topology_device.candidate_capacity;
        if (!device_rerank_result.has_value() && params.rerank.use_pq2_refine) {
            const std::size_t refine_top_l = std::min<std::size_t>(
                params.rerank.pq2_refine_top_l == 0 ? topology_device.candidate_capacity
                                                    : params.rerank.pq2_refine_top_l,
                topology_device.candidate_capacity);
            topoanns::detail::Pq2RefineBatchResult pq2_refine_result =
                topoanns::detail::RunPq2RefineBatchDevice(resources, topology_device,
                                                          batch_query_buffer, batch_queries,
                                                          refine_top_l);
            rerank_candidates = pq2_refine_result.candidate_buffer.CopyToHost();
            rerank_candidate_capacity = pq2_refine_result.candidate_capacity;
        }

        for (std::size_t local = 0; local < batch_queries; ++local) {
            const auto* feature_candidate_begin =
                host_candidates.data() + local * topology_device.candidate_capacity;
            const float* query = batch_query_buffer.data() + local * dim;
            topoanns::RerankQueryResult rerank_result;
            if (device_rerank_result.has_value()) {
                rerank_result = std::move(device_rerank_result->queries[local]);
            } else {
                const auto* rerank_candidate_begin =
                    rerank_candidates.data() + local * rerank_candidate_capacity;
                rerank_result = BuildCpuExactRerankResult(
                    rerank_candidate_begin,
                    std::min<std::size_t>(host_stats[local].valid_candidates,
                                          rerank_candidate_capacity),
                    params.rerank.top_n, query, dim, params.rerank.top_k, vector_reader);
            }
            snapshots[query_offset + local] =
                BuildSnapshot(feature_candidate_begin, topology_device.candidate_capacity,
                              host_stats[local], rerank_result, params.rerank.top_k,
                              params.topology.top_l,
                              prefix_override == 0U ? params.topology.top_l : prefix_override);
        }
        std::cout << "[learned_stop_dataset] top_l=" << params.topology.top_l
                  << " prefix=" << (prefix_override == 0U ? params.topology.top_l
                                                         : prefix_override)
                  << " finished queries " << (query_offset + batch_queries)
                  << " / " << num_queries << std::endl;
    }

    unsetenv("TOPOANNS_CANDIDATE_STOP_PREFIX");
    return snapshots;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        const topoanns::BamRuntimeConfig bam_runtime = topoanns::ResolveBamRuntimeConfig(
            args.bam_config_path, args.bam_overrides, args.allow_bam_controller_override);
        if (cudaSetDevice(static_cast<int>(bam_runtime.cuda_device)) != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed.");
        }
        RequireL40Device(bam_runtime.cuda_device, "topoanns_learned_stop_dataset");

        std::vector<float> queries;
        std::optional<IntMatrix> gt;
        std::uint32_t query_rows = 0;
        std::uint32_t query_dim = 0;
        if (!args.learn_bvecs.empty()) {
            std::cout << "[learned_stop_dataset] loading learn queries from "
                      << args.learn_bvecs << std::endl;
            const topoanns::BvecsMetadata metadata = topoanns::ReadBvecsMetadata(args.learn_bvecs);
            if (args.query_start + args.num_queries > metadata.num_vectors) {
                throw std::runtime_error("Requested query range exceeds learn.bvecs rows.");
            }
            queries = topoanns::ReadBvecsRangeAsFloat32(
                args.learn_bvecs, args.query_start, args.num_queries);
            query_rows = static_cast<std::uint32_t>(args.num_queries);
            query_dim = static_cast<std::uint32_t>(metadata.dim);
        } else {
            std::cout << "[learned_stop_dataset] loading query matrix from "
                      << args.query_bin << std::endl;
            const FloatMatrix query_matrix = LoadFloatMatrix(args.query_bin);
            if (args.query_start + args.num_queries > query_matrix.rows) {
                throw std::runtime_error("Requested query range exceeds query-bin rows.");
            }
            query_rows = static_cast<std::uint32_t>(args.num_queries);
            query_dim = query_matrix.cols;
            queries.resize(args.num_queries * static_cast<std::size_t>(query_dim));
            std::copy_n(query_matrix.values.begin() +
                            args.query_start * static_cast<std::size_t>(query_dim),
                        queries.size(), queries.begin());

            std::cout << "[learned_stop_dataset] loading GT from " << args.gt_bin << std::endl;
            IntMatrix gt_matrix = LoadIntMatrix(args.gt_bin);
            if (args.query_start + args.num_queries > gt_matrix.rows) {
                throw std::runtime_error("Requested query range exceeds gt-bin rows.");
            }
            if (gt_matrix.cols < args.top_k) {
                throw std::runtime_error("gt-bin has fewer than top-k columns.");
            }
            gt = std::move(gt_matrix);
        }

        std::cout << "[learned_stop_dataset] loading topology + pq index" << std::endl;
        topoanns::SearchResources resources =
            topoanns::SearchResources::FromTopologyFile(args.index_dir / "topology.bin");
        resources.LoadPqIndex(args.index_dir / "_pq_pivots.bin",
                              args.index_dir / "_pq_compressed.bin");
        if (args.use_pq2_refine) {
            std::cout << "[learned_stop_dataset] loading pq2 refine index" << std::endl;
            resources.LoadPq2Index(args.pq2_pivots, args.pq2_codes, args.pq2_error_bounds);
        }
        resources.LoadVectorStore(args.index_dir / "vectors.ssd");
        topoanns::BamVectorProviderOptions bam_options;
        topoanns::ApplyBamRuntimeConfig(bam_runtime, &bam_options);
        bam_options.device_offset_bytes = args.bam_device_offset_bytes;
        resources.AttachVectorPageProvider(std::make_shared<topoanns::BamVectorPageProvider>(
            args.index_dir / "vectors.ssd",
            resources.vector_store_layout().header_bytes(),
            resources.vector_store_layout().page_size_bytes(),
            bam_options));

        std::cout << "[learned_stop_dataset] loading RVQ model" << std::endl;
        topoanns::RvqModel rvq_model = topoanns::RvqModel::Load(args.rvq_model);
        if (rvq_model.dim() != query_dim || resources.vector_store_header().dim != query_dim) {
            throw std::runtime_error("Query dim, RVQ dim, and vector dim must match.");
        }
        HostVectorStoreReader vector_reader(resources.vector_store_path(),
                                            resources.vector_store_layout());

        std::cout << "[learned_stop_dataset] opening output csv" << std::endl;
        std::ofstream out(args.output_csv);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open output csv: " + args.output_csv.string());
        }
        out << "query_id,top_l,rerank_top_n,stage_index,prefix,label,"
               "expanded_nodes,visited_nodes,valid_candidates,"
               "best_unexpanded_rank,unexpanded_next64,unexpanded_next256,"
               "d1,dk,boundary,gap_prefix,gap32,topk_churn,delta_dk,delta_boundary,"
               "baseline_expanded_nodes,baseline_visited_nodes,"
               "stop_recall,full_recall,delta_recall\n";

        for (std::size_t config_idx = 0; config_idx < args.top_l_values.size(); ++config_idx) {
            topoanns::TopoAnnsSearchParams params;
            params.topology.top_k = std::max(args.top_k, args.rerank_top_n_values[config_idx]);
            params.topology.top_l = args.top_l_values[config_idx];
            params.topology.search_width = args.search_width;
            params.topology.max_expansions = args.max_expansions;
            params.rerank.top_k = args.top_k;
            params.rerank.top_n = args.rerank_top_n_values[config_idx];
            params.rerank.mode = topoanns::RerankExecutionMode::kPersistent;
            params.rerank.rank_tile_size = args.rerank_rank_tile_size;
            params.rerank.use_pq2_refine = args.use_pq2_refine;
            params.rerank.use_pq2_bound_filter = args.use_pq2_bound_filter;
            params.rerank.pq2_refine_top_l = args.top_l_values[config_idx];
            params.rerank.use_early_stop = args.use_early_stop;
            params.rerank.early_stop_min_prefix = args.rerank_early_stop_min_prefix;
            params.rerank.early_stop_patience_tiles = args.rerank_early_stop_patience_tiles;
            params.rerank.use_learned_stop = args.use_learned_stop;
            params.rerank.learned_stop_model_path = args.rerank_learned_stop_model;

            const std::vector<QuerySnapshot> baseline = RunConfiguration(
                resources, vector_reader, rvq_model, queries, args.num_queries, args.batch_size,
                args.rvq_entry_count, params, 0);
            const std::vector<std::uint32_t> ladder =
                BuildSparsePrefixLadder(args.top_l_values[config_idx]);
            std::vector<float> prev_dk(args.num_queries, 0.0f);
            std::vector<float> prev_boundary(args.num_queries, 0.0f);
            std::vector<std::uint32_t> prev_feature_topk_ids(
                args.num_queries * args.top_k, topoanns::kInvalidNodeId);
            std::vector<std::uint8_t> has_prev(args.num_queries, 0U);

            for (std::size_t stage_idx = 0; stage_idx < ladder.size(); ++stage_idx) {
                const std::uint32_t prefix = ladder[stage_idx];
                const std::vector<QuerySnapshot> stage = RunConfiguration(
                    resources, vector_reader, rvq_model, queries, args.num_queries, args.batch_size,
                    args.rvq_entry_count, params, prefix);
                for (std::size_t query_id = 0; query_id < args.num_queries; ++query_id) {
                    const QuerySnapshot& current = stage[query_id];
                    const QuerySnapshot& full = baseline[query_id];
                    const float delta_dk =
                        has_prev[query_id] != 0U ? (prev_dk[query_id] - current.dk) : 0.0f;
                    const float delta_boundary =
                        has_prev[query_id] != 0U ? (prev_boundary[query_id] - current.boundary)
                                                 : 0.0f;
                    const float topk_churn = ComputeTopKChurn(
                        current.feature_topk_ids,
                        prev_feature_topk_ids.data() + query_id * args.top_k,
                        args.top_k, has_prev[query_id] != 0U);
                    float stop_recall = -1.0f;
                    float full_recall = -1.0f;
                    float delta_recall = -1.0f;
                    if (gt.has_value()) {
                        const std::int32_t* gt_row =
                            gt->values.data() +
                            (args.query_start + query_id) * static_cast<std::size_t>(gt->cols);
                        stop_recall = RecallAtK(current, gt_row, args.top_k);
                        full_recall = RecallAtK(full, gt_row, args.top_k);
                        delta_recall = full_recall - stop_recall;
                    }
                    out << (args.query_start + query_id) << ','
                        << args.top_l_values[config_idx] << ','
                        << args.rerank_top_n_values[config_idx] << ','
                        << stage_idx << ','
                        << prefix << ','
                        << (TopKEquals(current, full) ? 0 : 1) << ','
                        << current.expanded_nodes << ','
                        << current.visited_nodes << ','
                        << current.valid_candidates << ','
                        << current.best_unexpanded_rank << ','
                        << current.unexpanded_next64 << ','
                        << current.unexpanded_next256 << ','
                        << current.d1 << ','
                        << current.dk << ','
                        << current.boundary << ','
                        << (current.boundary - current.dk) << ','
                        << current.gap32 << ','
                        << topk_churn << ','
                        << delta_dk << ','
                        << delta_boundary << ','
                        << full.expanded_nodes << ','
                        << full.visited_nodes << ','
                        << stop_recall << ','
                        << full_recall << ','
                        << delta_recall << '\n';
                    prev_dk[query_id] = current.dk;
                    prev_boundary[query_id] = current.boundary;
                    for (std::size_t i = 0; i < args.top_k; ++i) {
                        prev_feature_topk_ids[query_id * args.top_k + i] =
                            i < current.feature_topk_ids.size()
                                ? current.feature_topk_ids[i]
                                : topoanns::kInvalidNodeId;
                    }
                    has_prev[query_id] = 1U;
                }
                out.flush();
            }
        }

        std::cout << "[learned_stop_dataset] wrote " << args.output_csv << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_learned_stop_dataset] " << e.what() << std::endl;
        return 1;
    }
}
