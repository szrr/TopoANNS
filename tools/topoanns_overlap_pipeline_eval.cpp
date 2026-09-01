#include "topoanns/bam_runtime_config.hpp"
#include "topoanns/bam_vector_provider.hpp"
#include "topoanns/diskann_disk_index.hpp"
#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/rvq_entry_provider.hpp"
#include "topoanns/search_resources.hpp"
#include "../src/search/fused_rerank_device.hpp"
#include "../src/search/rerank_reuse_profiler.hpp"
#include "../src/search/topology_search_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

enum class PipelineMode {
    kSerial,
    kOverlap,
    kBoth,
};

bool g_bam_io_depth_profile = false;
bool g_topology_occupancy_profile = false;
bool g_topology_microbatch_io = false;
std::size_t g_topology_microbatch_queries = 2048;
std::size_t g_topology_microbatch_contexts = 3;
std::size_t g_topology_microbatch_io_blocks = 10;
std::size_t g_topology_microbatch_io_threads = 256;

bool DebugTopologyCache() {
    static const bool enabled = [] {
        const char* value = std::getenv("TOPOANNS_DEBUG_TOPOLOGY_CACHE");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

struct Args {
    std::filesystem::path index_dir;
    std::filesystem::path rvq_model;
    std::filesystem::path query_bin;
    std::filesystem::path gt_bin;
    std::filesystem::path base_pq_pivots;
    std::filesystem::path base_pq_codes;
    std::filesystem::path pq2_pivots;
    std::filesystem::path pq2_codes;
    std::filesystem::path pq2_error_bounds;
    std::filesystem::path hpq_base_pivots;
    std::filesystem::path hpq_outlier_pivots;
    std::filesystem::path hpq_codes;
    std::filesystem::path hpq_selector_bits;
    std::filesystem::path rerank_learned_stop_model;
    std::filesystem::path topology_learned_stop_model;
    std::size_t num_queries = 0;
    std::size_t top_k = 10;
    std::vector<std::size_t> top_l_values;
    std::vector<std::size_t> rerank_top_n_values;
    std::vector<std::size_t> microbatch_values;
    std::size_t serial_microbatch = 0;
    std::size_t rerank_rank_tile_size = 16;
    std::size_t search_width = 2;
    std::size_t max_expansions = 4096;
    std::size_t rvq_entry_count = 128;
    std::size_t ring_depth = 3;
    std::size_t warmup_runs = 0;
    std::size_t measured_runs = 1;
    bool rerank_use_early_stop = false;
    bool rerank_use_learned_stop = false;
    bool rerank_use_hpq = false;
    bool disable_pq2_refine = false;
    bool rerank_use_pq2_bound_filter = true;
    std::size_t rerank_early_stop_min_prefix = 32;
    std::size_t rerank_early_stop_patience_tiles = 1;
    bool traversal_use_heuristic_stop = false;
    std::size_t traversal_stop_divisor = 0;
    std::size_t traversal_stop_min_prefix = 32;
    std::size_t traversal_stop_prefix = 0;
    bool traversal_stop_use_expanded = false;
    std::string frontier_pq_mapping = "one-thread";
    std::size_t lut_prefetch_tile_chunks = 0;
    std::vector<std::size_t> lut_prefetch_tile_sweep_values;
    std::unordered_map<std::size_t, std::size_t> lut_prefetch_tile_by_top_l;
    bool merge_small_tail_batch = true;
    PipelineMode pipeline_mode = PipelineMode::kBoth;
    std::optional<std::filesystem::path> bam_config_path;
    topoanns::BamRuntimeConfigOverrides bam_overrides;
    bool allow_bam_controller_override = false;
    std::size_t bam_device_offset_bytes = 0;
    std::optional<std::size_t> topology_ssd_device_offset_bytes;
    std::filesystem::path combined_node_index;
    std::optional<std::size_t> combined_node_ssd_device_offset_bytes;
    bool enable_exact_reuse = false;
    bool exact_reuse_ab_sequence = false;
    std::size_t exact_reuse_cache_capacity = 0;
    bool bam_io_depth_profile = false;
    bool topology_occupancy_profile = false;
    std::optional<double> topology_cache_ratio;
    std::vector<double> topology_cache_ratio_values;
    std::optional<double> topology_cache_gb;
    std::filesystem::path topology_io_concurrency_trace;
    std::size_t topology_io_warmup_requests = 100000;
    std::size_t topology_io_measure_requests = 200000;
    std::size_t topology_io_blocks = 6;
    std::size_t topology_io_threads = 256;
    bool topology_microbatch_io = false;
    std::size_t topology_microbatch_queries = 2048;
    std::vector<std::size_t> topology_microbatch_query_values;
    std::size_t topology_microbatch_contexts = 3;
    std::size_t topology_microbatch_io_blocks = 10;
    std::vector<std::size_t> topology_microbatch_io_block_values;
    std::size_t topology_microbatch_io_threads = 256;
    std::filesystem::path expanded_trace_output_dir;
};

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

struct BatchSlice {
    std::size_t query_offset = 0;
    std::size_t num_queries = 0;
};

struct RerankControlConfig {
    bool use_early_stop = false;
    bool use_learned_stop = false;
    bool use_hpq = false;
    bool use_pq2_refine = true;
    bool use_pq2_bound_filter = true;
    std::size_t early_stop_min_prefix = 32;
    std::size_t early_stop_patience_tiles = 1;
    std::size_t rank_tile_size = 16;
    std::filesystem::path learned_stop_model;
};

struct ExactReuseControlConfig {
    bool enabled = false;
    std::uint32_t query_dim = 0;
    std::uint32_t node_bytes = 0;
    std::uint32_t nodes_per_page = 0;
    std::size_t cache_capacity = 0;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_overlap_pipeline_eval"
        << " --index-dir <path>"
        << " --rvq-model <path>"
        << " --query-bin <path>"
        << " --gt-bin <path>"
        << " [--base-pq-pivots <path> --base-pq-codes <path>]"
        << " --pq2-pivots <path>"
        << " --pq2-codes <path>"
        << " --pq2-error-bounds <path>"
        << " [--disable-pq2-refine]"
        << " [--rerank-use-hpq --hpq-base-pivots <path> --hpq-outlier-pivots <path>"
        << " --hpq-codes <path> --hpq-selector-bits <path>]"
        << " [--topology-learned-stop-model <path>]"
        << " [--rerank-use-early-stop]"
        << " [--rerank-use-learned-stop --rerank-learned-stop-model <path>]"
        << " [--rerank-disable-pq2-bound-filter]"
        << " [--rerank-early-stop-min-prefix <count>]"
        << " [--rerank-early-stop-patience-tiles <count>]"
        << " [--traversal-use-heuristic-stop]"
        << " [--traversal-stop-divisor <count>]"
        << " [--traversal-stop-min-prefix <count>]"
        << " [--traversal-stop-prefix <count>]"
        << " [--traversal-stop-use-expanded]"
        << " --num-queries <count>"
        << " --top-l-values <csv>"
        << " [--rerank-top-n-values <csv>]"
        << " --microbatch-values <csv>"
        << " [--serial-microbatch <count; 0 uses the overlap microbatch>]"
        << " [--pipeline-mode <serial|overlap|both>]"
        << " [--frontier-pq-mapping <one-thread>]"
        << " [--lut-prefetch-tile-chunks <count>]"
        << " [--lut-prefetch-tile-sweep-values <csv>]"
        << " [--lut-prefetch-tile-policy <top_l:tile,...>]"
        << " [--merge-small-tail-batch|--no-merge-small-tail-batch]"
        << " [--top-k <count>]"
        << " [--rerank-rank-tile-size <count>]"
        << " [--search-width <count>]"
        << " [--max-expansions <count>]"
        << " [--rvq-entry-count <count>]"
        << " [--ring-depth <count>]"
        << " [--warmup-runs <count>]"
        << " [--measured-runs <count>]"
        << " [--bam-config-path <path>]"
        << " [--allow-bam-controller-override]"
        << " [--bam-controller-path <path>]"
        << " [--bam-device-offset-bytes <bytes>]"
        << " [--topology-ssd-device-offset-bytes <bytes>]"
        << " [--enable-exact-reuse]"
        << " [--exact-reuse-ab-sequence]"
        << " [--combined-node-index <path>]"
        << " [--combined-node-ssd-device-offset-bytes <bytes>]"
        << " [--exact-reuse-cache-capacity <power-of-two>]"
        << " [--topology-cache-ratio <0..1>]"
        << " [--topology-cache-ratio-values <csv>]"
        << " [--topology-cache-gb <gb>]"
        << " [--bam-page-cache-bytes <bytes>]"
        << " [--bam-queue-depth <count>]"
        << " [--bam-num-queues <count>]"
        << " [--bam-cuda-device <id>]"
        << " [--bam-io-depth-profile]"
        << " [--topology-occupancy-profile]"
        << " [--topology-io-concurrency-trace <uint64-page-id-file>]"
        << " [--topology-io-warmup-requests <count>]"
        << " [--topology-io-measure-requests <count>]"
        << " [--topology-io-blocks <count>]"
        << " [--topology-io-threads <count>]"
        << " [--topology-microbatch-io]"
        << " [--topology-microbatch-queries <count>]"
        << " [--topology-microbatch-query-values <csv; 0 selects old path>]"
        << " [--topology-microbatch-contexts <count>]"
        << " [--topology-microbatch-io-blocks <count>]"
        << " [--topology-microbatch-io-block-values <csv>]"
        << " [--topology-microbatch-io-threads <count>]"
        << " [--expanded-trace-output-dir <path>]"
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
    if (values.empty()) {
        throw std::runtime_error("CSV list must not be empty.");
    }
    return values;
}

std::vector<double> ParseCsvDoubles(const std::string& text) {
    std::vector<double> values;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token =
            text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            values.push_back(std::stod(token));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (values.empty()) {
        throw std::runtime_error("CSV list must not be empty.");
    }
    return values;
}

PipelineMode ParsePipelineMode(const std::string& text) {
    if (text == "serial") {
        return PipelineMode::kSerial;
    }
    if (text == "overlap") {
        return PipelineMode::kOverlap;
    }
    if (text == "both") {
        return PipelineMode::kBoth;
    }
    throw std::runtime_error("Unsupported pipeline mode: " + text);
}

std::unordered_map<std::size_t, std::size_t> ParseTilePolicy(const std::string& text) {
    std::unordered_map<std::size_t, std::size_t> values;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token =
            text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            const std::size_t colon = token.find(':');
            if (colon == std::string::npos) {
                throw std::runtime_error(
                    "lut-prefetch-tile-policy entries must be formatted as top_l:tile");
            }
            const std::size_t top_l = std::stoull(token.substr(0, colon));
            const std::size_t tile = std::stoull(token.substr(colon + 1));
            values[top_l] = tile;
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (values.empty()) {
        throw std::runtime_error("lut-prefetch-tile-policy must not be empty.");
    }
    return values;
}

void ConfigureFrontierPqMapping(const std::string& frontier_pq_mapping,
                                std::size_t lut_prefetch_tile_chunks) {
    if (frontier_pq_mapping != "one-thread") {
        throw std::runtime_error(
            "Unsupported frontier-pq-mapping: " + frontier_pq_mapping +
            ". Only one-thread is supported.");
    }
    ::setenv("TOPOANNS_FRONTIER_PQ_MODE", "gustann", 1);
    ::unsetenv("TOPOANNS_FRONTIER_PQ_WARPS");
    if (lut_prefetch_tile_chunks == 0) {
        ::unsetenv("TOPOANNS_GUSTANN_LUT_TILE_CHUNKS");
    } else {
        ::setenv("TOPOANNS_GUSTANN_LUT_TILE_CHUNKS",
                 std::to_string(lut_prefetch_tile_chunks).c_str(), 1);
    }
}

void ConfigureTraversalStop(const Args& args) {
    ::unsetenv("TOPOANNS_CANDIDATE_STOP_DIVISOR");
    ::unsetenv("TOPOANNS_CANDIDATE_STOP_MIN_PREFIX");
    ::unsetenv("TOPOANNS_CANDIDATE_STOP_PREFIX");
    ::unsetenv("TOPOANNS_CANDIDATE_STOP_USE_EXPANDED");
    ::unsetenv("TOPOANNS_LEARNED_STOP_MODEL");

    if (!args.topology_learned_stop_model.empty()) {
        ::setenv("TOPOANNS_LEARNED_STOP_MODEL",
                 args.topology_learned_stop_model.c_str(), 1);
        return;
    }
    if (!args.traversal_use_heuristic_stop) {
        return;
    }
    if (args.traversal_stop_divisor > 1) {
        ::setenv("TOPOANNS_CANDIDATE_STOP_DIVISOR",
                 std::to_string(args.traversal_stop_divisor).c_str(), 1);
    }
    ::setenv("TOPOANNS_CANDIDATE_STOP_MIN_PREFIX",
             std::to_string(args.traversal_stop_min_prefix).c_str(), 1);
    if (args.traversal_stop_prefix != 0) {
        ::setenv("TOPOANNS_CANDIDATE_STOP_PREFIX",
                 std::to_string(args.traversal_stop_prefix).c_str(), 1);
    }
    if (args.traversal_stop_use_expanded) {
        ::setenv("TOPOANNS_CANDIDATE_STOP_USE_EXPANDED", "1", 1);
    }
}

std::vector<BatchSlice> BuildBatchPlan(std::size_t num_queries,
                                       std::size_t microbatch,
                                       bool merge_small_tail_batch) {
    if (microbatch == 0) {
        throw std::runtime_error("microbatch must be non-zero.");
    }
    std::vector<BatchSlice> plan;
    if (num_queries == 0) {
        return plan;
    }
    const std::size_t full_batches = num_queries / microbatch;
    const std::size_t remainder = num_queries % microbatch;
    std::size_t query_offset = 0;
    if (merge_small_tail_batch && remainder != 0 && full_batches >= 1) {
        for (std::size_t batch = 0; batch + 1 < full_batches; ++batch) {
            plan.push_back(BatchSlice{query_offset, microbatch});
            query_offset += microbatch;
        }
        plan.push_back(BatchSlice{query_offset, microbatch + remainder});
        return plan;
    }
    while (query_offset < num_queries) {
        const std::size_t batch_queries = std::min(microbatch, num_queries - query_offset);
        plan.push_back(BatchSlice{query_offset, batch_queries});
        query_offset += batch_queries;
    }
    return plan;
}

void RequireL40Device(std::uint32_t device_id, const char* context) {
    cudaDeviceProp props{};
    const cudaError_t status = cudaGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) +
                                 ": cudaGetDeviceProperties failed for requested BAM GPU.");
    }
    const std::string gpu_name(props.name);
    if (gpu_name.find("L40") == std::string::npos) {
        throw std::runtime_error(std::string(context) +
                                 ": BAM path must run on an NVIDIA L40, but the selected CUDA "
                                 "device is \"" + gpu_name + "\".");
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
        } else if (flag == "--query-bin") {
            args.query_bin = read_value("--query-bin");
        } else if (flag == "--gt-bin") {
            args.gt_bin = read_value("--gt-bin");
        } else if (flag == "--base-pq-pivots") {
            args.base_pq_pivots = read_value("--base-pq-pivots");
        } else if (flag == "--base-pq-codes") {
            args.base_pq_codes = read_value("--base-pq-codes");
        } else if (flag == "--pq2-pivots") {
            args.pq2_pivots = read_value("--pq2-pivots");
        } else if (flag == "--pq2-codes") {
            args.pq2_codes = read_value("--pq2-codes");
        } else if (flag == "--pq2-error-bounds") {
            args.pq2_error_bounds = read_value("--pq2-error-bounds");
        } else if (flag == "--hpq-base-pivots") {
            args.hpq_base_pivots = read_value("--hpq-base-pivots");
        } else if (flag == "--hpq-outlier-pivots") {
            args.hpq_outlier_pivots = read_value("--hpq-outlier-pivots");
        } else if (flag == "--hpq-codes") {
            args.hpq_codes = read_value("--hpq-codes");
        } else if (flag == "--hpq-selector-bits") {
            args.hpq_selector_bits = read_value("--hpq-selector-bits");
        } else if (flag == "--rerank-learned-stop-model") {
            args.rerank_learned_stop_model = read_value("--rerank-learned-stop-model");
        } else if (flag == "--rerank-use-early-stop") {
            args.rerank_use_early_stop = true;
        } else if (flag == "--rerank-use-learned-stop") {
            args.rerank_use_learned_stop = true;
        } else if (flag == "--rerank-use-hpq") {
            args.rerank_use_hpq = true;
        } else if (flag == "--disable-pq2-refine") {
            args.disable_pq2_refine = true;
        } else if (flag == "--rerank-disable-pq2-bound-filter") {
            args.rerank_use_pq2_bound_filter = false;
        } else if (flag == "--rerank-early-stop-min-prefix") {
            args.rerank_early_stop_min_prefix =
                std::stoull(read_value("--rerank-early-stop-min-prefix"));
        } else if (flag == "--rerank-early-stop-patience-tiles") {
            args.rerank_early_stop_patience_tiles =
                std::stoull(read_value("--rerank-early-stop-patience-tiles"));
        } else if (flag == "--topology-learned-stop-model") {
            args.topology_learned_stop_model = read_value("--topology-learned-stop-model");
        } else if (flag == "--traversal-use-heuristic-stop") {
            args.traversal_use_heuristic_stop = true;
        } else if (flag == "--traversal-stop-divisor") {
            args.traversal_stop_divisor =
                std::stoull(read_value("--traversal-stop-divisor"));
        } else if (flag == "--traversal-stop-min-prefix") {
            args.traversal_stop_min_prefix =
                std::stoull(read_value("--traversal-stop-min-prefix"));
        } else if (flag == "--traversal-stop-prefix") {
            args.traversal_stop_prefix =
                std::stoull(read_value("--traversal-stop-prefix"));
        } else if (flag == "--traversal-stop-use-expanded") {
            args.traversal_stop_use_expanded = true;
        } else if (flag == "--num-queries") {
            args.num_queries = std::stoull(read_value("--num-queries"));
        } else if (flag == "--top-k") {
            args.top_k = std::stoull(read_value("--top-k"));
        } else if (flag == "--rerank-rank-tile-size") {
            args.rerank_rank_tile_size = std::stoull(read_value("--rerank-rank-tile-size"));
        } else if (flag == "--top-l-values") {
            args.top_l_values = ParseCsvSizes(read_value("--top-l-values"));
        } else if (flag == "--rerank-top-n-values") {
            args.rerank_top_n_values = ParseCsvSizes(read_value("--rerank-top-n-values"));
        } else if (flag == "--microbatch-values") {
            args.microbatch_values = ParseCsvSizes(read_value("--microbatch-values"));
        } else if (flag == "--serial-microbatch") {
            args.serial_microbatch = std::stoull(read_value("--serial-microbatch"));
        } else if (flag == "--pipeline-mode") {
            args.pipeline_mode = ParsePipelineMode(read_value("--pipeline-mode"));
        } else if (flag == "--frontier-pq-mapping") {
            args.frontier_pq_mapping = read_value("--frontier-pq-mapping");
        } else if (flag == "--lut-prefetch-tile-chunks") {
            args.lut_prefetch_tile_chunks =
                std::stoull(read_value("--lut-prefetch-tile-chunks"));
        } else if (flag == "--lut-prefetch-tile-sweep-values") {
            args.lut_prefetch_tile_sweep_values =
                ParseCsvSizes(read_value("--lut-prefetch-tile-sweep-values"));
        } else if (flag == "--lut-prefetch-tile-policy") {
            args.lut_prefetch_tile_by_top_l =
                ParseTilePolicy(read_value("--lut-prefetch-tile-policy"));
        } else if (flag == "--merge-small-tail-batch") {
            args.merge_small_tail_batch = true;
        } else if (flag == "--no-merge-small-tail-batch") {
            args.merge_small_tail_batch = false;
        } else if (flag == "--search-width") {
            args.search_width = std::stoull(read_value("--search-width"));
        } else if (flag == "--max-expansions") {
            args.max_expansions = std::stoull(read_value("--max-expansions"));
        } else if (flag == "--rvq-entry-count") {
            args.rvq_entry_count = std::stoull(read_value("--rvq-entry-count"));
        } else if (flag == "--ring-depth") {
            args.ring_depth = std::stoull(read_value("--ring-depth"));
        } else if (flag == "--warmup-runs") {
            args.warmup_runs = std::stoull(read_value("--warmup-runs"));
        } else if (flag == "--measured-runs") {
            args.measured_runs = std::stoull(read_value("--measured-runs"));
        } else if (flag == "--bam-config-path") {
            args.bam_config_path = read_value("--bam-config-path");
        } else if (flag == "--allow-bam-controller-override") {
            args.allow_bam_controller_override = true;
        } else if (flag == "--bam-controller-path") {
            args.bam_overrides.controller_path = read_value("--bam-controller-path");
        } else if (flag == "--bam-device-offset-bytes") {
            args.bam_device_offset_bytes = std::stoull(read_value("--bam-device-offset-bytes"));
        } else if (flag == "--topology-ssd-device-offset-bytes") {
            args.topology_ssd_device_offset_bytes =
                std::stoull(read_value("--topology-ssd-device-offset-bytes"));
        } else if (flag == "--enable-exact-reuse") {
            args.enable_exact_reuse = true;
        } else if (flag == "--exact-reuse-ab-sequence") {
            args.exact_reuse_ab_sequence = true;
        } else if (flag == "--combined-node-index") {
            args.combined_node_index = read_value("--combined-node-index");
        } else if (flag == "--combined-node-ssd-device-offset-bytes") {
            args.combined_node_ssd_device_offset_bytes =
                std::stoull(read_value("--combined-node-ssd-device-offset-bytes"));
        } else if (flag == "--exact-reuse-cache-capacity") {
            args.exact_reuse_cache_capacity =
                std::stoull(read_value("--exact-reuse-cache-capacity"));
        } else if (flag == "--topology-cache-ratio") {
            args.topology_cache_ratio = std::stod(read_value("--topology-cache-ratio"));
        } else if (flag == "--topology-cache-ratio-values") {
            args.topology_cache_ratio_values =
                ParseCsvDoubles(read_value("--topology-cache-ratio-values"));
        } else if (flag == "--topology-cache-gb") {
            args.topology_cache_gb = std::stod(read_value("--topology-cache-gb"));
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
            args.bam_overrides.cuda_device =
                static_cast<std::uint32_t>(std::stoul(read_value("--bam-cuda-device")));
        } else if (flag == "--bam-io-depth-profile") {
            args.bam_io_depth_profile = true;
        } else if (flag == "--topology-occupancy-profile") {
            args.topology_occupancy_profile = true;
        } else if (flag == "--topology-io-concurrency-trace") {
            args.topology_io_concurrency_trace = read_value("--topology-io-concurrency-trace");
        } else if (flag == "--topology-io-warmup-requests") {
            args.topology_io_warmup_requests =
                std::stoull(read_value("--topology-io-warmup-requests"));
        } else if (flag == "--topology-io-measure-requests") {
            args.topology_io_measure_requests =
                std::stoull(read_value("--topology-io-measure-requests"));
        } else if (flag == "--topology-io-blocks") {
            args.topology_io_blocks =
                std::stoull(read_value("--topology-io-blocks"));
        } else if (flag == "--topology-io-threads") {
            args.topology_io_threads =
                std::stoull(read_value("--topology-io-threads"));
        } else if (flag == "--topology-microbatch-io") {
            args.topology_microbatch_io = true;
        } else if (flag == "--topology-microbatch-queries") {
            args.topology_microbatch_queries =
                std::stoull(read_value("--topology-microbatch-queries"));
        } else if (flag == "--topology-microbatch-query-values") {
            args.topology_microbatch_query_values =
                ParseCsvSizes(read_value("--topology-microbatch-query-values"));
            args.topology_microbatch_io = true;
        } else if (flag == "--topology-microbatch-contexts") {
            args.topology_microbatch_contexts =
                std::stoull(read_value("--topology-microbatch-contexts"));
        } else if (flag == "--topology-microbatch-io-blocks") {
            args.topology_microbatch_io_blocks =
                std::stoull(read_value("--topology-microbatch-io-blocks"));
        } else if (flag == "--topology-microbatch-io-block-values") {
            args.topology_microbatch_io_block_values =
                ParseCsvSizes(read_value("--topology-microbatch-io-block-values"));
            args.topology_microbatch_io = true;
        } else if (flag == "--topology-microbatch-io-threads") {
            args.topology_microbatch_io_threads =
                std::stoull(read_value("--topology-microbatch-io-threads"));
        } else if (flag == "--expanded-trace-output-dir") {
            args.expanded_trace_output_dir = read_value("--expanded-trace-output-dir");
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }
    if (args.index_dir.empty() || args.rvq_model.empty() || args.query_bin.empty() ||
        args.gt_bin.empty() || args.num_queries == 0 ||
        args.top_l_values.empty() || args.microbatch_values.empty()) {
        Usage();
    }
    if (args.base_pq_pivots.empty() != args.base_pq_codes.empty()) {
        throw std::runtime_error(
            "base-pq-pivots and base-pq-codes must be provided together.");
    }
    if (args.disable_pq2_refine && args.rerank_use_hpq) {
        throw std::runtime_error("disable-pq2-refine and rerank-use-hpq are mutually exclusive.");
    }
    if (args.rerank_use_hpq) {
        if (args.hpq_base_pivots.empty() || args.hpq_outlier_pivots.empty() ||
            args.hpq_codes.empty() ||
            args.hpq_selector_bits.empty()) {
            throw std::runtime_error(
                "rerank-use-hpq requires outlier pivots, hybrid codes, and selector bits.");
        }
        if (args.rerank_use_early_stop || args.rerank_use_learned_stop ||
            args.rerank_use_pq2_bound_filter) {
            throw std::runtime_error(
                "HPQ is mutually exclusive with heuristic/learned stop and PQ2 bounds.");
        }
    } else if (!args.disable_pq2_refine &&
               (args.pq2_pivots.empty() || args.pq2_codes.empty() ||
                args.pq2_error_bounds.empty())) {
        throw std::runtime_error("non-HPQ runs require PQ2 pivots, codes, and bounds.");
    }
    if (args.disable_pq2_refine) {
        args.rerank_use_pq2_bound_filter = false;
    }
    if (args.rerank_use_learned_stop && args.rerank_learned_stop_model.empty()) {
        throw std::runtime_error(
            "rerank-use-learned-stop requires rerank-learned-stop-model.");
    }
    if (args.rerank_use_early_stop && args.rerank_use_learned_stop) {
        throw std::runtime_error(
            "rerank heuristic and learned stop are mutually exclusive.");
    }
    if (args.traversal_use_heuristic_stop && !args.topology_learned_stop_model.empty()) {
        throw std::runtime_error(
            "traversal heuristic stop and topology learned stop are mutually exclusive.");
    }
    const std::size_t topology_cache_mode_count =
        static_cast<std::size_t>(args.topology_cache_ratio.has_value()) +
        static_cast<std::size_t>(!args.topology_cache_ratio_values.empty()) +
        static_cast<std::size_t>(args.topology_cache_gb.has_value());
    if (topology_cache_mode_count > 1) {
        throw std::runtime_error(
            "topology-cache-ratio, topology-cache-ratio-values, and topology-cache-gb "
            "are mutually exclusive.");
    }
    if (args.topology_cache_ratio.has_value() &&
        (*args.topology_cache_ratio < 0.0 || *args.topology_cache_ratio > 1.0)) {
        throw std::runtime_error("topology-cache-ratio must be in [0, 1].");
    }
    for (const double ratio : args.topology_cache_ratio_values) {
        if (ratio < 0.0 || ratio > 1.0) {
            throw std::runtime_error(
                "every topology-cache-ratio-values entry must be in [0, 1].");
        }
    }
    std::vector<double> unique_topology_cache_ratios;
    unique_topology_cache_ratios.reserve(args.topology_cache_ratio_values.size());
    for (const double ratio : args.topology_cache_ratio_values) {
        if (std::find(unique_topology_cache_ratios.begin(),
                      unique_topology_cache_ratios.end(),
                      ratio) == unique_topology_cache_ratios.end()) {
            unique_topology_cache_ratios.push_back(ratio);
        }
    }
    args.topology_cache_ratio_values = std::move(unique_topology_cache_ratios);
    if (args.topology_cache_gb.has_value() && *args.topology_cache_gb < 0.0) {
        throw std::runtime_error("topology-cache-gb must be non-negative.");
    }
    if (args.exact_reuse_cache_capacity != 0 &&
        (args.exact_reuse_cache_capacity & (args.exact_reuse_cache_capacity - 1U)) != 0) {
        throw std::runtime_error("exact-reuse-cache-capacity must be a power of two.");
    }
    if (args.exact_reuse_ab_sequence && !args.enable_exact_reuse) {
        throw std::runtime_error(
            "exact-reuse-ab-sequence requires enable-exact-reuse.");
    }
    if (!args.lut_prefetch_tile_sweep_values.empty() &&
        (!args.lut_prefetch_tile_by_top_l.empty() || args.lut_prefetch_tile_chunks != 0)) {
        throw std::runtime_error(
            "lut-prefetch-tile-sweep-values cannot be combined with lut-prefetch-tile-policy "
            "or a non-zero lut-prefetch-tile-chunks default.");
    }
    if (args.rerank_top_n_values.empty()) {
        args.rerank_top_n_values = args.top_l_values;
    } else if (args.rerank_top_n_values.size() == 1 && args.top_l_values.size() > 1) {
        args.rerank_top_n_values.assign(args.top_l_values.size(), args.rerank_top_n_values.front());
    } else if (args.rerank_top_n_values.size() != args.top_l_values.size()) {
        throw std::runtime_error(
            "rerank-top-n-values must be omitted, a single value, or match top-l-values in length.");
    }
    std::vector<std::pair<std::size_t, std::size_t>> sweep_pairs;
    sweep_pairs.reserve(args.top_l_values.size());
    for (std::size_t i = 0; i < args.top_l_values.size(); ++i) {
        const std::size_t top_l = args.top_l_values[i];
        const std::size_t rerank_top_n = args.rerank_top_n_values[i];
        if (top_l == 0 || rerank_top_n == 0) {
            throw std::runtime_error("top-l-values and rerank-top-n-values must be non-zero.");
        }
        if (rerank_top_n > top_l) {
            throw std::runtime_error(
                "rerank-top-n-values entries must be <= their corresponding top-l-values.");
        }
        sweep_pairs.emplace_back(top_l, rerank_top_n);
    }
    std::sort(sweep_pairs.begin(), sweep_pairs.end());
    sweep_pairs.erase(
        std::unique(sweep_pairs.begin(), sweep_pairs.end(),
                    [](const auto& lhs, const auto& rhs) { return lhs == rhs; }),
        sweep_pairs.end());
    args.top_l_values.clear();
    args.rerank_top_n_values.clear();
    args.top_l_values.reserve(sweep_pairs.size());
    args.rerank_top_n_values.reserve(sweep_pairs.size());
    for (const auto& [top_l, rerank_top_n] : sweep_pairs) {
        args.top_l_values.push_back(top_l);
        args.rerank_top_n_values.push_back(rerank_top_n);
    }
    std::sort(args.microbatch_values.begin(), args.microbatch_values.end());
    args.microbatch_values.erase(
        std::unique(args.microbatch_values.begin(), args.microbatch_values.end()),
        args.microbatch_values.end());
    if (args.measured_runs == 0) {
        throw std::runtime_error("measured-runs must be non-zero.");
    }
    if (args.bam_io_depth_profile && args.pipeline_mode == PipelineMode::kOverlap) {
        throw std::runtime_error(
            "bam-io-depth-profile requires serial or both pipeline mode.");
    }
    if (args.topology_microbatch_io) {
        if (!args.topology_microbatch_query_values.empty() &&
            std::none_of(args.topology_microbatch_query_values.begin(),
                         args.topology_microbatch_query_values.end(),
                         [](std::size_t value) { return value != 0; })) {
            throw std::runtime_error(
                "topology-microbatch-query-values requires at least one non-zero value.");
        }
        if (std::any_of(args.topology_microbatch_io_block_values.begin(),
                        args.topology_microbatch_io_block_values.end(),
                        [](std::size_t value) { return value == 0; })) {
            throw std::runtime_error(
                "topology-microbatch-io-block-values must be non-zero.");
        }
        if (args.pipeline_mode != PipelineMode::kSerial) {
            throw std::runtime_error(
                "topology-microbatch-io currently requires pipeline-mode=serial.");
        }
        if (args.topology_microbatch_queries == 0 ||
            args.topology_microbatch_contexts == 0 ||
            args.topology_microbatch_io_blocks == 0 ||
            args.topology_microbatch_io_threads == 0 ||
            args.topology_microbatch_io_threads > 256 ||
            args.topology_microbatch_io_threads % 32 != 0) {
            throw std::runtime_error("invalid topology microbatch dimensions.");
        }
    }
    if (!args.topology_io_concurrency_trace.empty()) {
        if (args.combined_node_index.empty() ||
            !args.combined_node_ssd_device_offset_bytes.has_value()) {
            throw std::runtime_error(
                "topology I/O concurrency benchmark requires combined-node-index and its SSD offset.");
        }
        if (args.top_l_values.size() != 1) {
            throw std::runtime_error(
                "topology I/O concurrency benchmark requires exactly one top-L value.");
        }
        if (args.topology_io_warmup_requests == 0 || args.topology_io_measure_requests == 0 ||
            args.topology_io_blocks == 0 || args.topology_io_threads == 0 ||
            args.topology_io_threads > 1024 || args.topology_io_threads % 32 != 0) {
            throw std::runtime_error("invalid topology I/O concurrency benchmark dimensions.");
        }
    }
    if (!args.expanded_trace_output_dir.empty()) {
        if (args.pipeline_mode != PipelineMode::kSerial || args.measured_runs != 1 ||
            args.microbatch_values.size() != 1 || args.exact_reuse_ab_sequence ||
            !args.lut_prefetch_tile_sweep_values.empty() || args.topology_microbatch_io) {
            throw std::runtime_error(
                "expanded trace requires serial mode, one measured run, one microbatch, "
                "one tile configuration, and the standard topology kernel.");
        }
        if (!args.topology_cache_ratio.has_value() || *args.topology_cache_ratio != 0.0 ||
            !args.topology_cache_ratio_values.empty() || args.topology_cache_gb.has_value()) {
            throw std::runtime_error(
                "expanded trace requires an explicit topology-cache-ratio=0.");
        }
        if (args.traversal_use_heuristic_stop ||
            !args.topology_learned_stop_model.empty() ||
            args.traversal_stop_divisor != 0 || args.traversal_stop_prefix != 0 ||
            args.traversal_stop_use_expanded) {
            throw std::runtime_error(
                "expanded trace requires traversal early stop to be fully disabled.");
        }
    }
    return args;
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    return alignment == 0 ? value : ((value + alignment - 1) / alignment) * alignment;
}

std::uint64_t ComputeTopologyCachedNodeCount(const topoanns::SearchResources& resources,
                                             const Args& args) {
    const std::uint64_t num_nodes = resources.num_nodes();
    if (args.topology_cache_ratio.has_value()) {
        return std::min<std::uint64_t>(
            num_nodes,
            static_cast<std::uint64_t>(*args.topology_cache_ratio *
                                       static_cast<double>(num_nodes)));
    }
    if (args.topology_cache_gb.has_value()) {
        constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
        const double cache_bytes = *args.topology_cache_gb * kBytesPerGiB;
        const double node_bytes =
            static_cast<double>(resources.degree()) * sizeof(std::uint32_t);
        return std::min<std::uint64_t>(
            num_nodes, static_cast<std::uint64_t>(cache_bytes / node_bytes));
    }
    return num_nodes;
}

struct TopologyCacheSweepPoint {
    double ratio = 1.0;
    std::uint64_t cached_node_count = 0;
};

std::vector<TopologyCacheSweepPoint> BuildTopologyCacheSweep(
    const topoanns::SearchResources& resources,
    const Args& args) {
    const std::uint64_t num_nodes = resources.num_nodes();
    std::vector<TopologyCacheSweepPoint> sweep;
    if (!args.topology_cache_ratio_values.empty()) {
        sweep.reserve(args.topology_cache_ratio_values.size());
        for (const double ratio : args.topology_cache_ratio_values) {
            sweep.push_back({
                ratio,
                std::min<std::uint64_t>(
                    num_nodes,
                    static_cast<std::uint64_t>(
                        ratio * static_cast<double>(num_nodes))),
            });
        }
        return sweep;
    }
    const std::uint64_t cached_node_count =
        ComputeTopologyCachedNodeCount(resources, args);
    sweep.push_back({
        num_nodes == 0
            ? 0.0
            : static_cast<double>(cached_node_count) /
                  static_cast<double>(num_nodes),
        cached_node_count,
    });
    return sweep;
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

std::vector<std::uint64_t> LoadPageTrace(const std::filesystem::path& path) {
    const std::uintmax_t bytes = std::filesystem::file_size(path);
    if (bytes == 0 || bytes % sizeof(std::uint64_t) != 0) {
        throw std::runtime_error("page trace must contain packed uint64 page IDs.");
    }
    std::vector<std::uint64_t> trace(bytes / sizeof(std::uint64_t));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(trace.data()),
               static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("failed to read page trace: " + path.string());
    }
    return trace;
}

struct StageAOutput {
    std::size_t query_offset = 0;
    std::size_t num_queries = 0;
    std::vector<float> host_queries;
    topoanns::CudaBuffer<float> device_queries;
    topoanns::CudaBuffer<topoanns::detail::DeviceTopologySearchStats> stats_buffer;
    topoanns::CudaBuffer<topoanns::detail::DeviceTopologyProfileCycles> profile_buffer;
    std::size_t expanded_trace_stride = 0;
    topoanns::CudaBuffer<std::uint32_t> expanded_trace_buffer;
    topoanns::detail::DeviceTopologyBatchResult rerank_topology;
    std::size_t rerank_ssd_records_per_page = 0;
    std::size_t rerank_ssd_record_stride_bytes = 0;
    double stage_a_wall_ms = 0.0;
    double stage_a_query_copy_ms = 0.0;
    double stage_a_query_upload_ms = 0.0;
    double stage_a_entry_batch_slice_ms = 0.0;
    double stage_a_pq_query_tables_ms = 0.0;
    double stage_a_topology_stage_ms = 0.0;
    double stage_a_pq2_refine_stage_ms = 0.0;
    double topology_kernel_ms = 0.0;
    double pq2_query_tables_ms = 0.0;
    double pq2_refine_kernel_ms = 0.0;
    std::size_t occupancy_dynamic_shared_bytes = 0;
    std::size_t occupancy_blocks_per_sm = 0;
    std::size_t occupancy_sm_count = 0;
    std::size_t occupancy_resident_blocks = 0;
    std::size_t occupancy_max_io_warps = 0;
    bool topology_microbatch_enabled = false;
    topoanns::detail::TopologyMicrobatchExecutionProfile topology_microbatch_profile;
};

std::uint64_t CounterDelta(std::uint64_t after, std::uint64_t before) {
    return after >= before ? after - before : 0;
}

std::size_t DepthQuantile(const std::vector<std::uint64_t>& histogram,
                          double quantile) {
    std::uint64_t total = 0;
    for (const std::uint64_t count : histogram) {
        total += count;
    }
    if (total == 0) {
        return 0;
    }
    const std::uint64_t target =
        static_cast<std::uint64_t>(quantile * static_cast<double>(total - 1));
    std::uint64_t cumulative = 0;
    for (std::size_t depth = 0; depth < histogram.size(); ++depth) {
        cumulative += histogram[depth];
        if (cumulative > target) {
            return depth;
        }
    }
    return histogram.size() - 1;
}

void PrintStageABamProfile(const topoanns::BamIoProfileSnapshot& before,
                           const topoanns::BamIoProfileSnapshot& after,
                           const StageAOutput& stage_a) {
    long double weighted_depth = 0.0;
    std::uint64_t depth_samples = 0;
    std::size_t max_depth = 0;
    for (std::size_t depth = 0; depth < after.submit_depth_histogram.size(); ++depth) {
        const std::uint64_t count = after.submit_depth_histogram[depth];
        weighted_depth += static_cast<long double>(depth) * count;
        depth_samples += count;
        if (count != 0) {
            max_depth = depth;
        }
    }
    const double mean_depth =
        depth_samples == 0
            ? 0.0
            : static_cast<double>(weighted_depth / depth_samples);
    std::cout << "[topoanns_stage_a_bam_profile]"
              << " query_offset=" << stage_a.query_offset
              << " num_queries=" << stage_a.num_queries
              << " dynamic_shared_bytes=" << stage_a.occupancy_dynamic_shared_bytes
              << " blocks_per_sm=" << stage_a.occupancy_blocks_per_sm
              << " sm_count=" << stage_a.occupancy_sm_count
              << " resident_ctas=" << stage_a.occupancy_resident_blocks
              << " theoretical_max_outstanding=" << stage_a.occupancy_max_io_warps
              << " gpu_cache_hits="
              << CounterDelta(after.gpu_cache_hits, before.gpu_cache_hits)
              << " host_cache_hits="
              << CounterDelta(after.host_cache_hits, before.host_cache_hits)
              << " physical_reads="
              << CounterDelta(after.physical_reads, before.physical_reads)
              << " submissions="
              << CounterDelta(after.profiled_submissions, before.profiled_submissions)
              << " completions="
              << CounterDelta(after.profiled_completions, before.profiled_completions)
              << " final_outstanding=" << after.current_outstanding
              << " mean_depth=" << mean_depth
              << " p50_depth=" << DepthQuantile(after.submit_depth_histogram, 0.50)
              << " p90_depth=" << DepthQuantile(after.submit_depth_histogram, 0.90)
              << " p99_depth=" << DepthQuantile(after.submit_depth_histogram, 0.99)
              << " max_depth=" << max_depth << std::endl;
    if (stage_a.topology_microbatch_enabled) {
        const auto& profile = stage_a.topology_microbatch_profile;
        std::cout << "[topoanns_topology_microbatch_profile]"
                  << " query_offset=" << stage_a.query_offset
                  << " num_queries=" << stage_a.num_queries
                  << " wall_ms=" << profile.wall_ms
                  << " summed_io_kernel_ms=" << profile.summed_io_kernel_ms
                  << " logical_io_requests=" << profile.logical_io_requests
                  << " io_batches=" << profile.io_batches
                  << " nonempty_io_batches=" << profile.nonempty_io_batches
                  << " min_nonempty_batch_requests="
                  << profile.min_nonempty_batch_requests
                  << " max_batch_requests=" << profile.max_batch_requests
                  << " validation_mismatch_neighbors="
                  << profile.validation_mismatch_neighbors
                  << std::endl;
    }
}

struct BatchOutcome {
    std::size_t query_offset = 0;
    std::size_t num_queries = 0;
    std::size_t result_top_n = 0;
    std::vector<std::uint32_t> final_node_ids;
    topoanns::CudaBuffer<topoanns::detail::DeviceTopologySearchStats> stats_buffer;
    topoanns::CudaBuffer<topoanns::detail::DeviceTopologyProfileCycles> profile_buffer;
    std::size_t expanded_trace_stride = 0;
    topoanns::CudaBuffer<std::uint32_t> expanded_trace_buffer;
    double stage_a_wall_ms = 0.0;
    double stage_a_query_copy_ms = 0.0;
    double stage_a_query_upload_ms = 0.0;
    double stage_a_entry_batch_slice_ms = 0.0;
    double stage_a_pq_query_tables_ms = 0.0;
    double stage_a_topology_stage_ms = 0.0;
    double stage_a_pq2_refine_stage_ms = 0.0;
    double stage_b_wall_ms = 0.0;
    double topology_kernel_ms = 0.0;
    std::size_t occupancy_dynamic_shared_bytes = 0;
    std::size_t occupancy_blocks_per_sm = 0;
    std::size_t occupancy_sm_count = 0;
    double pq2_query_tables_ms = 0.0;
    double pq2_refine_kernel_ms = 0.0;
    double rerank_kernel_ms = 0.0;
    double topology_learned_stop_model_ms = 0.0;
    double topology_learned_stop_feature_ms = 0.0;
    double topology_learned_stop_find_first_set_ms = 0.0;
    double topology_learned_stop_count_bits_ms = 0.0;
    double topology_learned_stop_topk_churn_ms = 0.0;
    double topology_learned_stop_logit_eval_ms = 0.0;
    double learned_stop_model_ms = 0.0;
    double learned_stop_logit_eval_ms = 0.0;
    double rerank_query_block_ms = 0.0;
    std::size_t exact_count = 0;
    std::size_t rerank_ssd_io_pages = 0;
    std::size_t reused_exact_count = 0;
    double exact_reuse_lookup_ms = 0.0;
    double combined_node_read_ms = 0.0;
    double exact_reuse_insert_ms = 0.0;
};

struct ExperimentMetrics {
    std::size_t top_l = 0;
    std::size_t rerank_top_n = 0;
    std::size_t microbatch = 0;
    double topology_cache_ratio = 1.0;
    std::uint64_t topology_cached_node_count = 0;
    std::size_t num_queries = 0;
    std::size_t top_k = 0;
    double rvq_precompute_ms = 0.0;
    double pipeline_ms = 0.0;
    double end_to_end_ms = 0.0;
    double pipeline_qps = 0.0;
    double end_to_end_qps = 0.0;
    double stage_a_wall_ms = 0.0;
    double stage_a_query_copy_ms = 0.0;
    double stage_a_query_upload_ms = 0.0;
    double stage_a_entry_batch_slice_ms = 0.0;
    double stage_a_pq_query_tables_ms = 0.0;
    double stage_a_topology_stage_ms = 0.0;
    double stage_a_pq2_refine_stage_ms = 0.0;
    double stage_b_wall_ms = 0.0;
    double overlap_efficiency = 0.0;
    double topology_kernel_ms = 0.0;
    double avg_query_pq_ms = 0.0;
    double avg_query_pq_compute_ms = 0.0;
    double avg_query_pq_prefetch_issue_ms = 0.0;
    double avg_query_pq_prefetch_wait_ms = 0.0;
    double avg_query_queue_ms = 0.0;
    double pq_runtime_fraction = 0.0;
    std::size_t occupancy_dynamic_shared_bytes = 0;
    std::size_t occupancy_blocks_per_sm = 0;
    std::size_t occupancy_sm_count = 0;
    double pq2_query_tables_ms = 0.0;
    double pq2_refine_kernel_ms = 0.0;
    double rerank_kernel_ms = 0.0;
    double topology_learned_stop_model_ms = 0.0;
    double topology_learned_stop_feature_ms = 0.0;
    double topology_learned_stop_find_first_set_ms = 0.0;
    double topology_learned_stop_count_bits_ms = 0.0;
    double topology_learned_stop_topk_churn_ms = 0.0;
    double topology_learned_stop_logit_eval_ms = 0.0;
    double learned_stop_model_ms = 0.0;
    double learned_stop_logit_eval_ms = 0.0;
    double rerank_query_block_ms = 0.0;
    double rerank_bw_gbps = 0.0;
    double recall_at_10 = 0.0;
    std::uint64_t result_id_checksum = 0;
    double avg_expanded = 0.0;
    double avg_visited = 0.0;
    double avg_io_pages = 0.0;
    double avg_rerank_ssd_io_pages = 0.0;
    double avg_reused_exact = 0.0;
    double avg_exact_reuse_inserts = 0.0;
    double avg_exact_reuse_overflows = 0.0;
    double exact_reuse_lookup_ms = 0.0;
    double combined_node_read_ms = 0.0;
    double exact_reuse_insert_ms = 0.0;
    double avg_topology_io_pages = 0.0;
    double avg_total_io_pages = 0.0;
    std::size_t batch_count = 0;
    std::size_t first_batch_queries = 0;
    std::size_t last_batch_queries = 0;
    bool tail_batch_merged = false;
    std::size_t lut_prefetch_tile_chunks = 0;
};

enum class RunKind {
    kWarmup,
    kMeasured,
};

const char* RunKindLabel(RunKind kind) {
    return kind == RunKind::kWarmup ? "warmup" : "measured";
}

template <typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}

    void Push(T value) {
        std::unique_lock<std::mutex> lock(mu_);
        not_full_.wait(lock, [&] { return closed_ || queue_.size() < capacity_; });
        if (closed_) {
            throw std::runtime_error("Push on closed queue.");
        }
        queue_.push_back(std::move(value));
        not_empty_.notify_one();
    }

    bool Pop(T* out) {
        std::unique_lock<std::mutex> lock(mu_);
        not_empty_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        *out = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return true;
    }

    void Close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    std::size_t capacity_ = 0;
    std::mutex mu_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::deque<T> queue_;
    bool closed_ = false;
};

std::vector<float> SliceHostQueries(const FloatMatrix& queries,
                                    std::size_t query_offset,
                                    std::size_t num_queries) {
    const std::size_t dim = queries.cols;
    const float* begin = queries.values.data() + query_offset * dim;
    const float* end = begin + num_queries * dim;
    return std::vector<float>(begin, end);
}

topoanns::CudaBuffer<float> UploadQueriesToDeviceAsync(const std::vector<float>& host_queries,
                                                       cudaStream_t stream) {
    topoanns::CudaBuffer<float> device_queries =
        topoanns::CudaBuffer<float>::Allocate(host_queries.size());
    if (!host_queries.empty()) {
        topoanns::ThrowIfCudaError(
            cudaMemcpyAsync(device_queries.get(), host_queries.data(),
                            host_queries.size() * sizeof(float),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsyncHostToDevice(queries)");
    }
    return device_queries;
}

topoanns::DeviceEntryBatch SliceDeviceEntryBatch(const topoanns::DeviceEntryBatch& full_batch,
                                                 std::size_t query_offset,
                                                 std::size_t num_queries,
                                                 cudaStream_t stream) {
    topoanns::DeviceEntryBatch batch;
    batch.num_queries = num_queries;
    batch.entries_per_query = full_batch.entries_per_query;
    const std::size_t id_count = num_queries * full_batch.entries_per_query;
    std::vector<std::uint32_t> host_offsets(num_queries + 1, 0);
    for (std::size_t i = 0; i <= num_queries; ++i) {
        host_offsets[i] = static_cast<std::uint32_t>(i * full_batch.entries_per_query);
    }
    batch.offsets = topoanns::CudaBuffer<std::uint32_t>::Allocate(host_offsets.size());
    if (!host_offsets.empty()) {
        topoanns::ThrowIfCudaError(
            cudaMemcpyAsync(batch.offsets.get(), host_offsets.data(),
                            host_offsets.size() * sizeof(std::uint32_t),
                            cudaMemcpyHostToDevice, stream),
            "cudaMemcpyAsyncHostToDevice(entry_offsets)");
    }
    batch.ids = topoanns::CudaBuffer<std::uint32_t>::Allocate(id_count);
    if (id_count != 0) {
        const std::size_t id_offset = query_offset * full_batch.entries_per_query;
        topoanns::ThrowIfCudaError(
            cudaMemcpyAsync(batch.ids.get(), full_batch.ids.get() + id_offset,
                            id_count * sizeof(std::uint32_t),
                            cudaMemcpyDeviceToDevice, stream),
            "cudaMemcpyAsyncDeviceToDevice(entry_ids)");
    }
    return batch;
}

struct ScopedCudaEvent {
    cudaEvent_t event = nullptr;

    ScopedCudaEvent() {
        topoanns::ThrowIfCudaError(cudaEventCreate(&event), "cudaEventCreate");
    }

    ~ScopedCudaEvent() {
        if (event != nullptr) {
            cudaEventDestroy(event);
        }
    }

    ScopedCudaEvent(const ScopedCudaEvent&) = delete;
    ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;
};

double ElapsedEventMs(cudaEvent_t begin, cudaEvent_t end) {
    float ms = 0.0f;
    topoanns::ThrowIfCudaError(cudaEventElapsedTime(&ms, begin, end), "cudaEventElapsedTime");
    return static_cast<double>(ms);
}

double RecallAtKFromIds(const std::vector<BatchOutcome>& outcomes,
                        const IntMatrix& gt,
                        std::size_t top_k) {
    std::size_t matched = 0;
    std::size_t total_queries = 0;
    for (const auto& outcome : outcomes) {
        for (std::size_t local_q = 0; local_q < outcome.num_queries; ++local_q) {
            const std::int32_t* gt_row =
                gt.values.data() +
                (outcome.query_offset + local_q) * static_cast<std::size_t>(gt.cols);
            const std::uint32_t* found =
                outcome.final_node_ids.data() + local_q * outcome.result_top_n;
            for (std::size_t i = 0; i < top_k; ++i) {
                for (std::size_t rank = 0; rank < top_k; ++rank) {
                    if (static_cast<std::int32_t>(found[i]) == gt_row[rank]) {
                        ++matched;
                        break;
                    }
                }
            }
        }
        total_queries += outcome.num_queries;
    }
    return total_queries == 0
               ? 0.0
               : static_cast<double>(matched) /
                     static_cast<double>(total_queries * top_k);
}

std::uint64_t ResultIdChecksum(const std::vector<BatchOutcome>& outcomes) {
    std::uint64_t checksum = 1469598103934665603ULL;
    for (const auto& outcome : outcomes) {
        for (const std::uint32_t node_id : outcome.final_node_ids) {
            checksum ^= static_cast<std::uint64_t>(node_id);
            checksum *= 1099511628211ULL;
        }
    }
    return checksum;
}

StageAOutput RunStageA(const topoanns::SearchResources& resources,
                       const FloatMatrix& queries,
                       const topoanns::DeviceEntryBatch& full_entry_batch,
                       std::size_t query_offset,
                       std::size_t batch_queries,
                       std::size_t top_l,
                       std::size_t top_k,
                       std::size_t search_width,
                       std::size_t max_expansions,
                       std::uint64_t topology_cached_node_count,
                       const RerankControlConfig& rerank_control,
                       const ExactReuseControlConfig& exact_reuse_control,
                       bool enable_expanded_trace,
                       cudaStream_t stream) {
    const auto begin = std::chrono::steady_clock::now();
    StageAOutput output;
    ScopedCudaEvent query_upload_begin;
    ScopedCudaEvent query_upload_end;
    ScopedCudaEvent entry_batch_begin;
    ScopedCudaEvent entry_batch_end;
    ScopedCudaEvent pq_tables_begin;
    ScopedCudaEvent pq_tables_end;
    ScopedCudaEvent topology_begin;
    ScopedCudaEvent topology_end;
    ScopedCudaEvent pq2_refine_begin;
    ScopedCudaEvent pq2_refine_end;
    output.query_offset = query_offset;
    output.num_queries = batch_queries;
    if (exact_reuse_control.enabled &&
        topology_cached_node_count < resources.num_nodes()) {
        output.rerank_ssd_records_per_page = exact_reuse_control.nodes_per_page;
        output.rerank_ssd_record_stride_bytes = exact_reuse_control.node_bytes;
    }
    const std::size_t dim = queries.cols;
    const auto query_copy_begin = std::chrono::steady_clock::now();
    output.host_queries.resize(batch_queries * dim);
    const float* begin_ptr = queries.values.data() + query_offset * dim;
    std::copy(begin_ptr, begin_ptr + batch_queries * dim, output.host_queries.begin());
    const auto query_copy_end = std::chrono::steady_clock::now();
    output.stage_a_query_copy_ms =
        std::chrono::duration<double, std::milli>(query_copy_end - query_copy_begin).count();

    topoanns::ThrowIfCudaError(cudaEventRecord(query_upload_begin.event, stream),
                               "cudaEventRecord(stageA_query_upload_begin)");
    output.device_queries = UploadQueriesToDeviceAsync(output.host_queries, stream);
    topoanns::ThrowIfCudaError(cudaEventRecord(query_upload_end.event, stream),
                               "cudaEventRecord(stageA_query_upload_end)");

    topoanns::ThrowIfCudaError(cudaEventRecord(entry_batch_begin.event, stream),
                               "cudaEventRecord(stageA_entry_batch_begin)");
    topoanns::DeviceEntryBatch entry_batch =
        SliceDeviceEntryBatch(full_entry_batch, query_offset, batch_queries, stream);
    topoanns::ThrowIfCudaError(cudaEventRecord(entry_batch_end.event, stream),
                               "cudaEventRecord(stageA_entry_batch_end)");

    topoanns::ThrowIfCudaError(cudaEventRecord(pq_tables_begin.event, stream),
                               "cudaEventRecord(stageA_pq_tables_begin)");
    topoanns::PqQueryDistanceTables query_tables =
        topoanns::PqQueryDistanceTables::FromFloatQueriesDeviceBufferAsync(
            resources.pq_index(), output.device_queries, batch_queries, stream);
    topoanns::ThrowIfCudaError(cudaEventRecord(pq_tables_end.event, stream),
                               "cudaEventRecord(stageA_pq_tables_end)");
    const topoanns::PqDistanceOracle oracle(
        resources.pq_index(), std::move(query_tables));

    topoanns::TopologySearchParams topology_params;
    topology_params.top_k = top_k;
    topology_params.top_l = top_l;
    topology_params.candidate_queue_size = top_l;
    topology_params.search_width = search_width;
    topology_params.max_expansions = max_expansions;
    topology_params.topology_cached_node_count = topology_cached_node_count;
    topology_params.enable_exact_reuse = exact_reuse_control.enabled;
    topology_params.exact_reuse_device_queries = output.device_queries.get();
    topology_params.exact_reuse_query_dim = exact_reuse_control.query_dim;
    topology_params.combined_node_bytes = exact_reuse_control.node_bytes;
    topology_params.combined_nodes_per_page = exact_reuse_control.nodes_per_page;
    topology_params.exact_reuse_cache_capacity = exact_reuse_control.cache_capacity;
    topology_params.enable_occupancy_profile =
        g_bam_io_depth_profile || g_topology_occupancy_profile;
    topology_params.enable_expanded_trace = enable_expanded_trace;

    topoanns::ThrowIfCudaError(cudaEventRecord(topology_begin.event, stream),
                               "cudaEventRecord(stageA_topology_begin)");
    if (DebugTopologyCache()) {
        std::cerr << "[topology_cache_host_debug] before_topology_kernel"
                  << " query_offset=" << query_offset
                  << " batch_queries=" << batch_queries
                  << " top_l=" << top_l
                  << " cached_node_count=" << topology_cached_node_count << '\n';
    }
    topoanns::detail::DeviceTopologyBatchResult topology_result;
    if (g_topology_microbatch_io) {
        topoanns::ThrowIfCudaError(
            cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(before topology microbatch)");
        const topoanns::detail::TopologyMicrobatchExecutionConfig execution{
            g_topology_microbatch_queries,
            g_topology_microbatch_contexts,
            g_topology_microbatch_io_blocks,
            g_topology_microbatch_io_threads,
        };
        output.topology_microbatch_enabled = true;
        topology_result =
            topoanns::detail::RunTopologySearchKernelBatchDeviceMicrobatched(
                resources, oracle, entry_batch, topology_params, execution,
                &output.topology_microbatch_profile);
    } else {
        topology_result = topoanns::detail::RunTopologySearchKernelBatchDevice(
            resources, oracle, entry_batch, topology_params, stream);
    }
    if (DebugTopologyCache()) {
        std::cerr << "[topology_cache_host_debug] after_topology_call"
                  << " kernel_ms=" << topology_result.kernel_ms << '\n';
    }
    topoanns::ThrowIfCudaError(cudaEventRecord(topology_end.event, stream),
                               "cudaEventRecord(stageA_topology_end)");
    output.topology_kernel_ms = topology_result.kernel_ms;
    output.occupancy_dynamic_shared_bytes =
        topology_result.occupancy_dynamic_shared_bytes;
    output.occupancy_blocks_per_sm = topology_result.occupancy_blocks_per_sm;
    output.occupancy_sm_count = topology_result.occupancy_sm_count;
    output.occupancy_resident_blocks = topology_result.occupancy_resident_blocks;
    output.occupancy_max_io_warps = topology_result.occupancy_max_io_warps;

    topoanns::ThrowIfCudaError(cudaEventRecord(pq2_refine_begin.event, stream),
                               "cudaEventRecord(stageA_pq2_refine_begin)");
    if (DebugTopologyCache()) {
        std::cerr << "[topology_cache_host_debug] before_pq2_refine\n";
    }
    if (rerank_control.use_pq2_refine) {
        topoanns::detail::Pq2RefineBatchResult pq2_refine =
            rerank_control.use_hpq
                ? topoanns::detail::RunHpqRefineBatchDevice(
                      resources, topology_result, output.device_queries, batch_queries, top_l, stream)
                : topoanns::detail::RunPq2RefineBatchDevice(
                      resources, topology_result, output.device_queries, batch_queries, top_l, stream);
        if (DebugTopologyCache()) {
            std::cerr << "[topology_cache_host_debug] after_pq2_refine_call"
                      << " kernel_ms=" << pq2_refine.kernel_ms << '\n';
        }
        output.pq2_query_tables_ms = pq2_refine.pq_profile.total_ms;
        output.pq2_refine_kernel_ms = pq2_refine.kernel_ms;
        output.rerank_topology.num_queries = pq2_refine.num_queries;
        output.rerank_topology.candidate_capacity = pq2_refine.candidate_capacity;
        output.rerank_topology.candidate_buffer = std::move(pq2_refine.candidate_buffer);
    } else {
        output.rerank_topology.num_queries = topology_result.num_queries;
        output.rerank_topology.candidate_capacity = topology_result.candidate_capacity;
        output.rerank_topology.candidate_buffer = std::move(topology_result.candidate_buffer);
    }
    topoanns::ThrowIfCudaError(cudaEventRecord(pq2_refine_end.event, stream),
                               "cudaEventRecord(stageA_pq2_refine_end)");
    output.rerank_topology.exact_reuse_cache_capacity =
        topology_result.exact_reuse_cache_capacity;
    output.expanded_trace_stride = topology_result.expanded_trace_stride;
    output.expanded_trace_buffer =
        std::move(topology_result.expanded_trace_buffer);
    output.rerank_topology.exact_reuse_node_ids =
        std::move(topology_result.exact_reuse_node_ids);
    output.rerank_topology.exact_reuse_distances =
        std::move(topology_result.exact_reuse_distances);
    output.stats_buffer = std::move(topology_result.stats_buffer);
    output.profile_buffer = std::move(topology_result.profile_buffer);
    if (DebugTopologyCache()) {
        std::cerr << "[topology_cache_host_debug] before_stage_a_sync\n";
    }
    topoanns::ThrowIfCudaError(cudaEventSynchronize(pq2_refine_end.event),
                               "cudaEventSynchronize(stageA_pq2_refine_end)");
    if (rerank_control.use_pq2_refine &&
        topoanns::detail::RerankReuseProfilingEnabled()) {
        topoanns::detail::ProfileRerankReuseCandidates(
            topology_result.candidate_buffer,
            topology_result.candidate_capacity,
            output.rerank_topology.candidate_buffer,
            output.rerank_topology.num_queries,
            output.rerank_topology.candidate_capacity,
            top_l,
            resources.num_nodes());
    }
    if (DebugTopologyCache()) {
        std::cerr << "[topology_cache_host_debug] after_stage_a_sync\n";
    }
    output.stage_a_query_upload_ms =
        ElapsedEventMs(query_upload_begin.event, query_upload_end.event);
    output.stage_a_entry_batch_slice_ms =
        ElapsedEventMs(entry_batch_begin.event, entry_batch_end.event);
    output.stage_a_pq_query_tables_ms =
        ElapsedEventMs(pq_tables_begin.event, pq_tables_end.event);
    output.stage_a_topology_stage_ms =
        ElapsedEventMs(topology_begin.event, topology_end.event);
    output.stage_a_pq2_refine_stage_ms =
        ElapsedEventMs(pq2_refine_begin.event, pq2_refine_end.event);
    const auto end = std::chrono::steady_clock::now();
    output.stage_a_wall_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    return output;
}

BatchOutcome RunStageB(const topoanns::SearchResources& resources,
                       StageAOutput input,
                       std::size_t top_k,
                       std::size_t top_l,
                       std::size_t rerank_top_n,
                       const RerankControlConfig& rerank_control,
                       cudaStream_t stream) {
    const auto begin = std::chrono::steady_clock::now();
    topoanns::RerankExactParams rerank_params;
    rerank_params.top_k = top_k;
    rerank_params.top_n = rerank_top_n;
    rerank_params.mode = topoanns::RerankExecutionMode::kPersistent;
    rerank_params.rank_tile_size = rerank_control.rank_tile_size;
    rerank_params.use_pq2_refine = false;
    rerank_params.use_pq2_bound_filter = rerank_control.use_pq2_bound_filter;
    rerank_params.pq2_refine_top_l = top_l;
    rerank_params.use_early_stop = rerank_control.use_early_stop;
    rerank_params.use_learned_stop = rerank_control.use_learned_stop;
    rerank_params.learned_stop_model_path = rerank_control.learned_stop_model;
    rerank_params.ssd_records_per_page = input.rerank_ssd_records_per_page;
    rerank_params.ssd_record_stride_bytes = input.rerank_ssd_record_stride_bytes;
    rerank_params.early_stop_min_prefix = rerank_control.early_stop_min_prefix;
    rerank_params.early_stop_patience_tiles = rerank_control.early_stop_patience_tiles;

    if (DebugTopologyCache()) {
        std::cerr << "[topology_cache_host_debug] before_stage_b_rerank"
                  << " query_offset=" << input.query_offset
                  << " num_queries=" << input.num_queries
                  << " top_l=" << top_l
                  << " rerank_top_n=" << rerank_top_n << '\n';
    }
    topoanns::detail::PersistentBamRerankRunResult rerank_result =
        topoanns::detail::RunPersistentBamRerankFloat32(
            resources, input.device_queries, input.host_queries, input.rerank_topology,
            input.num_queries, rerank_params, stream);
    if (DebugTopologyCache()) {
        std::cerr << "[topology_cache_host_debug] after_stage_b_rerank_call"
                  << " kernel_ms=" << rerank_result.kernel_ms << '\n';
    }
    std::vector<std::uint32_t> final_node_ids = rerank_result.final_node_ids.CopyToHost();
    if (DebugTopologyCache()) {
        std::cerr << "[topology_cache_host_debug] after_stage_b_copy\n";
    }
    const auto end = std::chrono::steady_clock::now();
    BatchOutcome outcome;
    outcome.query_offset = input.query_offset;
    outcome.num_queries = input.num_queries;
    outcome.result_top_n = rerank_result.result_top_n;
    outcome.final_node_ids = std::move(final_node_ids);
    outcome.stats_buffer = std::move(input.stats_buffer);
    outcome.profile_buffer = std::move(input.profile_buffer);
    outcome.expanded_trace_stride = input.expanded_trace_stride;
    outcome.expanded_trace_buffer =
        std::move(input.expanded_trace_buffer);
    outcome.stage_a_wall_ms = input.stage_a_wall_ms;
    outcome.stage_a_query_copy_ms = input.stage_a_query_copy_ms;
    outcome.stage_a_query_upload_ms = input.stage_a_query_upload_ms;
    outcome.stage_a_entry_batch_slice_ms = input.stage_a_entry_batch_slice_ms;
    outcome.stage_a_pq_query_tables_ms = input.stage_a_pq_query_tables_ms;
    outcome.stage_a_topology_stage_ms = input.stage_a_topology_stage_ms;
    outcome.stage_a_pq2_refine_stage_ms = input.stage_a_pq2_refine_stage_ms;
    outcome.stage_b_wall_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    outcome.topology_kernel_ms = input.topology_kernel_ms;
    outcome.occupancy_dynamic_shared_bytes = input.occupancy_dynamic_shared_bytes;
    outcome.occupancy_blocks_per_sm = input.occupancy_blocks_per_sm;
    outcome.occupancy_sm_count = input.occupancy_sm_count;
    outcome.pq2_query_tables_ms = input.pq2_query_tables_ms;
    outcome.pq2_refine_kernel_ms = input.pq2_refine_kernel_ms;
    outcome.rerank_kernel_ms = rerank_result.kernel_ms;
    outcome.learned_stop_model_ms = rerank_result.learned_stop_model_ms;
    outcome.learned_stop_logit_eval_ms = rerank_result.learned_stop_logit_eval_ms;
    outcome.rerank_query_block_ms = rerank_result.query_block_ms;
    outcome.exact_count = rerank_result.exact_count;
    outcome.rerank_ssd_io_pages = rerank_result.rerank_ssd_io_pages;
    outcome.reused_exact_count = rerank_result.reused_exact_count;
    outcome.exact_reuse_lookup_ms = rerank_result.exact_reuse_lookup_ms;
    return outcome;
}

void FinalizeMetrics(ExperimentMetrics* metrics,
                     std::vector<BatchOutcome>* outcomes,
                     const IntMatrix& gt,
                     std::uint32_t device_id) {
    std::sort(outcomes->begin(), outcomes->end(),
              [](const BatchOutcome& lhs, const BatchOutcome& rhs) {
                  return lhs.query_offset < rhs.query_offset;
              });
    topoanns::ThrowIfCudaError(cudaSetDevice(static_cast<int>(device_id)), "cudaSetDevice");
    cudaDeviceProp props;
    topoanns::ThrowIfCudaError(cudaGetDeviceProperties(&props, static_cast<int>(device_id)),
                               "cudaGetDeviceProperties");
    const double cycles_per_ms = static_cast<double>(props.clockRate);
    std::size_t total_expanded = 0;
    std::size_t total_visited = 0;
    std::size_t total_topology_io_pages = 0;
    std::size_t total_exact = 0;
    std::size_t total_rerank_ssd_io_pages = 0;
    std::size_t total_reused_exact = 0;
    std::size_t total_exact_reuse_inserts = 0;
    std::size_t total_exact_reuse_overflows = 0;
    for (auto& outcome : *outcomes) {
        const std::vector<topoanns::detail::DeviceTopologySearchStats> host_stats =
            outcome.stats_buffer.CopyToHost();
        const std::vector<topoanns::detail::DeviceTopologyProfileCycles> host_profile =
            outcome.profile_buffer.CopyToHost();
        for (const auto& stats : host_stats) {
            total_expanded += stats.expanded_nodes;
            total_visited += stats.visited_nodes;
            total_topology_io_pages += stats.topology_io_pages;
            total_exact_reuse_inserts += stats.exact_reuse_inserts;
            total_exact_reuse_overflows += stats.exact_reuse_overflows;
        }
        for (const auto& query_profile : host_profile) {
            metrics->avg_query_pq_ms +=
                static_cast<double>(query_profile.pq_cycles) / cycles_per_ms;
            metrics->avg_query_pq_compute_ms +=
                static_cast<double>(query_profile.pq_compute_cycles) / cycles_per_ms;
            metrics->avg_query_pq_prefetch_issue_ms +=
                static_cast<double>(query_profile.pq_prefetch_issue_cycles) / cycles_per_ms;
            metrics->avg_query_pq_prefetch_wait_ms +=
                static_cast<double>(query_profile.pq_prefetch_wait_cycles) / cycles_per_ms;
            metrics->avg_query_queue_ms +=
                static_cast<double>(query_profile.queue_cycles) / cycles_per_ms;
            outcome.topology_learned_stop_model_ms +=
                static_cast<double>(query_profile.learned_stop_model_cycles) / cycles_per_ms;
            outcome.topology_learned_stop_feature_ms +=
                static_cast<double>(query_profile.learned_stop_feature_cycles) / cycles_per_ms;
            outcome.topology_learned_stop_find_first_set_ms +=
                static_cast<double>(query_profile.learned_stop_find_first_set_cycles) /
                cycles_per_ms;
            outcome.topology_learned_stop_count_bits_ms +=
                static_cast<double>(query_profile.learned_stop_count_bits_cycles) / cycles_per_ms;
            outcome.topology_learned_stop_topk_churn_ms +=
                static_cast<double>(query_profile.learned_stop_topk_churn_cycles) /
                cycles_per_ms;
            outcome.topology_learned_stop_logit_eval_ms +=
                static_cast<double>(query_profile.learned_stop_logit_eval_cycles) /
                cycles_per_ms;
            outcome.combined_node_read_ms +=
                static_cast<double>(query_profile.combined_node_read_cycles) / cycles_per_ms;
            outcome.exact_reuse_insert_ms +=
                static_cast<double>(query_profile.exact_reuse_insert_cycles) / cycles_per_ms;
        }
        metrics->stage_a_wall_ms += outcome.stage_a_wall_ms;
        metrics->stage_a_query_copy_ms += outcome.stage_a_query_copy_ms;
        metrics->stage_a_query_upload_ms += outcome.stage_a_query_upload_ms;
        metrics->stage_a_entry_batch_slice_ms += outcome.stage_a_entry_batch_slice_ms;
        metrics->stage_a_pq_query_tables_ms += outcome.stage_a_pq_query_tables_ms;
        metrics->stage_a_topology_stage_ms += outcome.stage_a_topology_stage_ms;
        metrics->stage_a_pq2_refine_stage_ms += outcome.stage_a_pq2_refine_stage_ms;
        metrics->stage_b_wall_ms += outcome.stage_b_wall_ms;
        metrics->topology_kernel_ms += outcome.topology_kernel_ms;
        metrics->occupancy_dynamic_shared_bytes =
            std::max(metrics->occupancy_dynamic_shared_bytes,
                     outcome.occupancy_dynamic_shared_bytes);
        metrics->occupancy_blocks_per_sm =
            std::max(metrics->occupancy_blocks_per_sm, outcome.occupancy_blocks_per_sm);
        metrics->occupancy_sm_count =
            std::max(metrics->occupancy_sm_count, outcome.occupancy_sm_count);
        metrics->pq2_query_tables_ms += outcome.pq2_query_tables_ms;
        metrics->pq2_refine_kernel_ms += outcome.pq2_refine_kernel_ms;
        metrics->rerank_kernel_ms += outcome.rerank_kernel_ms;
        metrics->topology_learned_stop_model_ms += outcome.topology_learned_stop_model_ms;
        metrics->topology_learned_stop_feature_ms += outcome.topology_learned_stop_feature_ms;
        metrics->topology_learned_stop_find_first_set_ms +=
            outcome.topology_learned_stop_find_first_set_ms;
        metrics->topology_learned_stop_count_bits_ms +=
            outcome.topology_learned_stop_count_bits_ms;
        metrics->topology_learned_stop_topk_churn_ms +=
            outcome.topology_learned_stop_topk_churn_ms;
        metrics->topology_learned_stop_logit_eval_ms +=
            outcome.topology_learned_stop_logit_eval_ms;
        metrics->learned_stop_model_ms += outcome.learned_stop_model_ms;
        metrics->learned_stop_logit_eval_ms += outcome.learned_stop_logit_eval_ms;
        metrics->rerank_query_block_ms += outcome.rerank_query_block_ms;
        total_exact += outcome.exact_count;
        metrics->exact_reuse_lookup_ms += outcome.exact_reuse_lookup_ms;
        metrics->combined_node_read_ms += outcome.combined_node_read_ms;
        metrics->exact_reuse_insert_ms += outcome.exact_reuse_insert_ms;
        total_rerank_ssd_io_pages += outcome.rerank_ssd_io_pages;
        total_reused_exact += outcome.reused_exact_count;
    }
    const double query_count = static_cast<double>(metrics->num_queries);
    metrics->avg_query_pq_ms /= query_count;
    metrics->avg_query_pq_compute_ms /= query_count;
    metrics->avg_query_pq_prefetch_issue_ms /= query_count;
    metrics->avg_query_pq_prefetch_wait_ms /= query_count;
    metrics->avg_query_queue_ms /= query_count;
    const double profiled_query_ms =
        metrics->avg_query_pq_ms + metrics->avg_query_queue_ms;
    metrics->pq_runtime_fraction =
        profiled_query_ms == 0.0 ? 0.0 : metrics->avg_query_pq_ms / profiled_query_ms;
    metrics->avg_expanded =
        static_cast<double>(total_expanded) / static_cast<double>(metrics->num_queries);
    metrics->avg_visited =
        static_cast<double>(total_visited) / static_cast<double>(metrics->num_queries);
    metrics->avg_io_pages =
        static_cast<double>(total_exact) / static_cast<double>(metrics->num_queries);
    metrics->avg_rerank_ssd_io_pages =
        static_cast<double>(total_rerank_ssd_io_pages) /
        static_cast<double>(metrics->num_queries);
    metrics->avg_reused_exact =
        static_cast<double>(total_reused_exact) / static_cast<double>(metrics->num_queries);
    metrics->avg_exact_reuse_inserts =
        static_cast<double>(total_exact_reuse_inserts) /
        static_cast<double>(metrics->num_queries);
    metrics->avg_exact_reuse_overflows =
        static_cast<double>(total_exact_reuse_overflows) /
        static_cast<double>(metrics->num_queries);
    metrics->avg_topology_io_pages =
        static_cast<double>(total_topology_io_pages) / static_cast<double>(metrics->num_queries);
    metrics->avg_total_io_pages =
        metrics->avg_rerank_ssd_io_pages + metrics->avg_topology_io_pages;
    metrics->recall_at_10 = RecallAtKFromIds(*outcomes, gt, metrics->top_k);
    metrics->result_id_checksum = ResultIdChecksum(*outcomes);
    if (metrics->rerank_kernel_ms > 0.0) {
        metrics->rerank_bw_gbps =
            static_cast<double>(total_rerank_ssd_io_pages) * 4096.0 /
            (metrics->rerank_kernel_ms / 1000.0) / 1.0e9;
    }
    metrics->pipeline_qps =
        static_cast<double>(metrics->num_queries) / (metrics->pipeline_ms / 1000.0);
    metrics->end_to_end_ms = metrics->rvq_precompute_ms + metrics->pipeline_ms;
    metrics->end_to_end_qps =
        static_cast<double>(metrics->num_queries) / (metrics->end_to_end_ms / 1000.0);
}
void WriteExpandedTrace(const ExperimentMetrics& metrics,
                        std::vector<BatchOutcome>* outcomes,
                        const std::filesystem::path& output_dir) {
    if (output_dir.empty()) {
        return;
    }
    std::filesystem::create_directories(output_dir);
    const std::string suffix = "_L" + std::to_string(metrics.top_l) + ".csv";
    const std::filesystem::path trace_path = output_dir / ("expanded_trace" + suffix);
    const std::filesystem::path frequency_path =
        output_dir / ("expanded_frequency" + suffix);
    const std::filesystem::path summary_path =
        output_dir / ("expanded_hotset_summary" + suffix);

    std::ofstream trace_out(trace_path);
    if (!trace_out.is_open()) {
        throw std::runtime_error("failed to open expanded trace output.");
    }
    trace_out << "query_id,expanded_vertex_ids\n";
    std::vector<std::uint32_t> accesses;
    accesses.reserve(static_cast<std::size_t>(
        metrics.avg_expanded * static_cast<double>(metrics.num_queries) + 0.5));

    for (auto& outcome : *outcomes) {
        if (outcome.expanded_trace_stride == 0 ||
            outcome.expanded_trace_buffer.empty()) {
            throw std::runtime_error("expanded trace buffer is missing.");
        }
        const auto host_stats = outcome.stats_buffer.CopyToHost();
        const auto host_trace = outcome.expanded_trace_buffer.CopyToHost();
        if (host_stats.size() != outcome.num_queries ||
            host_trace.size() != outcome.num_queries * outcome.expanded_trace_stride) {
            throw std::runtime_error("expanded trace buffer shape is invalid.");
        }
        for (std::size_t local_query = 0; local_query < outcome.num_queries; ++local_query) {
            const std::size_t expanded_count = host_stats[local_query].expanded_nodes;
            if (expanded_count > outcome.expanded_trace_stride) {
                throw std::runtime_error("expanded trace overflow.");
            }
            trace_out << (outcome.query_offset + local_query);
            const std::size_t trace_offset =
                local_query * outcome.expanded_trace_stride;
            for (std::size_t i = 0; i < expanded_count; ++i) {
                const std::uint32_t node_id = host_trace[trace_offset + i];
                trace_out << ',' << node_id;
                accesses.push_back(node_id);
            }
            trace_out << '\n';
        }
    }
    trace_out.close();

    const std::size_t expected_accesses = static_cast<std::size_t>(
        metrics.avg_expanded * static_cast<double>(metrics.num_queries) + 0.5);
    if (accesses.size() != expected_accesses) {
        throw std::runtime_error("expanded trace count does not match aggregate statistics.");
    }
    std::sort(accesses.begin(), accesses.end());
    std::vector<std::pair<std::uint32_t, std::uint64_t>> frequencies;
    for (std::size_t begin = 0; begin < accesses.size();) {
        std::size_t end = begin + 1;
        while (end < accesses.size() && accesses[end] == accesses[begin]) {
            ++end;
        }
        frequencies.emplace_back(accesses[begin], end - begin);
        begin = end;
    }
    std::sort(frequencies.begin(), frequencies.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second != rhs.second ? lhs.second > rhs.second : lhs.first < rhs.first;
    });

    std::ofstream frequency_out(frequency_path);
    frequency_out << "rank,vertex_id,expand_count,vertex_coverage_pct,"
                     "cumulative_expansion_pct\n";
    std::vector<std::uint64_t> cumulative(frequencies.size());
    std::uint64_t running = 0;
    for (std::size_t i = 0; i < frequencies.size(); ++i) {
        running += frequencies[i].second;
        cumulative[i] = running;
        frequency_out << (i + 1) << ',' << frequencies[i].first << ','
                      << frequencies[i].second << ',' << std::setprecision(10)
                      << (100.0 * static_cast<double>(i + 1) /
                          static_cast<double>(frequencies.size()))
                      << ',' << (100.0 * static_cast<double>(running) /
                                 static_cast<double>(accesses.size())) << '\n';
    }

    std::ofstream summary_out(summary_path);
    summary_out << "top_vertex_pct,cumulative_expansion_pct\n";
    for (const std::size_t pct : {1U, 5U, 10U, 20U, 30U, 50U}) {
        const std::size_t count = std::min(
            frequencies.size(),
            (pct * frequencies.size() + 99U) / 100U);
        const std::uint64_t covered = count == 0 ? 0 : cumulative[count - 1];
        summary_out << pct << ',' << std::setprecision(10)
                    << (100.0 * static_cast<double>(covered) /
                        static_cast<double>(accesses.size())) << '\n';
    }
    std::cout << "[expanded_trace]"
              << " top_l=" << metrics.top_l
              << " queries=" << metrics.num_queries
              << " accesses=" << accesses.size()
              << " unique_vertices=" << frequencies.size()
              << " trace=" << trace_path
              << " frequency=" << frequency_path
              << " summary=" << summary_path << std::endl;
}


void PopulateBatchPlanMetrics(ExperimentMetrics* metrics,
                              const std::vector<BatchSlice>& batch_plan,
                              std::size_t total_queries,
                              std::size_t microbatch,
                              bool merge_small_tail_batch) {
    metrics->batch_count = batch_plan.size();
    if (!batch_plan.empty()) {
        metrics->first_batch_queries = batch_plan.front().num_queries;
        metrics->last_batch_queries = batch_plan.back().num_queries;
    }
    const std::size_t remainder = total_queries % microbatch;
    const std::size_t full_batches = total_queries / microbatch;
    metrics->tail_batch_merged =
        merge_small_tail_batch && remainder != 0 && full_batches >= 1;
}

ExperimentMetrics RunSerialExperiment(const topoanns::SearchResources& resources,
                                      const FloatMatrix& queries,
                                      const IntMatrix& gt,
                                      const topoanns::DeviceEntryBatch& full_entry_batch,
                                      std::size_t top_l,
                                      std::size_t rerank_top_n,
                                      std::size_t microbatch,
                                      std::size_t top_k,
                                      std::size_t search_width,
                                      std::size_t max_expansions,
                                      std::uint64_t topology_cached_node_count,
                                      const ExactReuseControlConfig& exact_reuse_control,
                                      const RerankControlConfig& rerank_control,
                                      double rvq_precompute_ms,
                                      std::uint32_t device_id,
                                      bool merge_small_tail_batch,
                                      topoanns::BamVectorPageProvider* bam_profile_provider,
                                      const std::filesystem::path& expanded_trace_output_dir) {
    topoanns::ThrowIfCudaError(cudaSetDevice(static_cast<int>(device_id)), "cudaSetDevice");
    cudaStream_t stream = nullptr;
    topoanns::ThrowIfCudaError(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                               "cudaStreamCreateWithFlags");
    ExperimentMetrics metrics;
    metrics.top_l = top_l;
    metrics.rerank_top_n = rerank_top_n;
    metrics.microbatch = microbatch;
    metrics.num_queries = gt.rows;
    metrics.top_k = top_k;
    metrics.rvq_precompute_ms = rvq_precompute_ms;
    std::vector<BatchOutcome> outcomes;
    const std::vector<BatchSlice> batch_plan =
        BuildBatchPlan(metrics.num_queries, microbatch, merge_small_tail_batch);
    PopulateBatchPlanMetrics(&metrics, batch_plan, metrics.num_queries, microbatch,
                             merge_small_tail_batch);
    const auto begin = std::chrono::steady_clock::now();
    for (const BatchSlice& batch : batch_plan) {
        topoanns::BamIoProfileSnapshot before_profile;
        if (bam_profile_provider != nullptr) {
            bam_profile_provider->ResetIoDepthProfile(true);
            before_profile = bam_profile_provider->ReadIoProfile();
        }
        StageAOutput stage_a = RunStageA(
            resources, queries, full_entry_batch, batch.query_offset,
            batch.num_queries, top_l, top_k, search_width,
            max_expansions, topology_cached_node_count, rerank_control, exact_reuse_control,
            !expanded_trace_output_dir.empty(), stream);
        if (bam_profile_provider != nullptr) {
            const topoanns::BamIoProfileSnapshot after_profile =
                bam_profile_provider->ReadIoProfile();
            PrintStageABamProfile(before_profile, after_profile, stage_a);
            bam_profile_provider->ResetIoDepthProfile(false);
        }
        outcomes.push_back(RunStageB(resources, std::move(stage_a), top_k, top_l,
                                     rerank_top_n,
                                     rerank_control, stream));
    }
    topoanns::ThrowIfCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy");
    const auto end = std::chrono::steady_clock::now();
    metrics.pipeline_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    FinalizeMetrics(&metrics, &outcomes, gt, device_id);
    WriteExpandedTrace(metrics, &outcomes, expanded_trace_output_dir);
    return metrics;
}

ExperimentMetrics RunOverlapExperiment(const topoanns::SearchResources& resources,
                                       const FloatMatrix& queries,
                                       const IntMatrix& gt,
                                       const topoanns::DeviceEntryBatch& full_entry_batch,
                                       std::size_t top_l,
                                       std::size_t rerank_top_n,
                                       std::size_t microbatch,
                                       std::size_t top_k,
                                       std::size_t search_width,
                                       std::size_t max_expansions,
                                       std::uint64_t topology_cached_node_count,
                                       const ExactReuseControlConfig& exact_reuse_control,
                                       const RerankControlConfig& rerank_control,
                                       double rvq_precompute_ms,
                                       std::uint32_t device_id,
                                       std::size_t ring_depth,
                                       bool merge_small_tail_batch) {
    ExperimentMetrics metrics;
    metrics.top_l = top_l;
    metrics.rerank_top_n = rerank_top_n;
    metrics.microbatch = microbatch;
    metrics.num_queries = gt.rows;
    metrics.top_k = top_k;
    metrics.rvq_precompute_ms = rvq_precompute_ms;
    const std::vector<BatchSlice> batch_plan =
        BuildBatchPlan(metrics.num_queries, microbatch, merge_small_tail_batch);
    PopulateBatchPlanMetrics(&metrics, batch_plan, metrics.num_queries, microbatch,
                             merge_small_tail_batch);

    BoundedQueue<StageAOutput> queue(ring_depth);
    std::vector<BatchOutcome> outcomes;
    std::mutex outcomes_mu;
    std::exception_ptr worker_error;
    std::mutex error_mu;

    const auto set_worker_error = [&](std::exception_ptr eptr) {
        std::lock_guard<std::mutex> lock(error_mu);
        if (!worker_error) {
            worker_error = eptr;
        }
    };

    const auto begin = std::chrono::steady_clock::now();
    std::thread stage_a_thread([&] {
        try {
            topoanns::ThrowIfCudaError(cudaSetDevice(static_cast<int>(device_id)), "cudaSetDevice");
            cudaStream_t stream = nullptr;
            topoanns::ThrowIfCudaError(
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags(stage_a)");
            for (const BatchSlice& batch : batch_plan) {
                queue.Push(RunStageA(resources, queries, full_entry_batch, batch.query_offset,
                                     batch.num_queries, top_l, top_k, search_width,
                                     max_expansions, topology_cached_node_count, rerank_control,
                                     exact_reuse_control, false, stream));
            }
            topoanns::ThrowIfCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy(stage_a)");
            queue.Close();
        } catch (...) {
            set_worker_error(std::current_exception());
            queue.Close();
        }
    });

    std::thread stage_b_thread([&] {
        try {
            topoanns::ThrowIfCudaError(cudaSetDevice(static_cast<int>(device_id)), "cudaSetDevice");
            cudaStream_t stream = nullptr;
            topoanns::ThrowIfCudaError(
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags(stage_b)");
            StageAOutput task;
            while (queue.Pop(&task)) {
                BatchOutcome outcome =
                    RunStageB(resources, std::move(task), top_k, top_l, rerank_top_n,
                              rerank_control, stream);
                std::lock_guard<std::mutex> lock(outcomes_mu);
                outcomes.push_back(std::move(outcome));
            }
            topoanns::ThrowIfCudaError(cudaStreamDestroy(stream), "cudaStreamDestroy(stage_b)");
        } catch (...) {
            set_worker_error(std::current_exception());
            queue.Close();
        }
    });

    stage_a_thread.join();
    stage_b_thread.join();
    if (worker_error) {
        std::rethrow_exception(worker_error);
    }
    const auto end = std::chrono::steady_clock::now();
    metrics.pipeline_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    FinalizeMetrics(&metrics, &outcomes, gt, device_id);
    const double min_stage = std::min(metrics.stage_a_wall_ms, metrics.stage_b_wall_ms);
    if (min_stage > 0.0) {
        metrics.overlap_efficiency =
            (metrics.stage_a_wall_ms + metrics.stage_b_wall_ms - metrics.pipeline_ms) /
            min_stage;
    }
    return metrics;
}

void PrintMetrics(const char* mode,
                  const char* exact_reuse_variant,
                  const ExperimentMetrics& metrics,
                  const std::optional<ExperimentMetrics>& baseline,
                  RunKind run_kind,
                  std::size_t repeat_id) {
    const double gap =
        std::max(metrics.stage_a_wall_ms, metrics.stage_b_wall_ms) > 0.0
            ? metrics.pipeline_ms /
                  std::max(metrics.stage_a_wall_ms, metrics.stage_b_wall_ms)
            : 0.0;
    const double overlap_ratio =
        std::min(metrics.stage_a_wall_ms, metrics.stage_b_wall_ms) > 0.0
            ? std::max(metrics.stage_a_wall_ms, metrics.stage_b_wall_ms) /
                  std::min(metrics.stage_a_wall_ms, metrics.stage_b_wall_ms)
            : 0.0;
    std::cout << "[topoanns_overlap_pipeline] mode=" << mode
              << " exact_reuse_variant=" << exact_reuse_variant
              << " run_kind=" << RunKindLabel(run_kind)
              << " repeat_id=" << repeat_id
              << " top_l=" << metrics.top_l
              << " rerank_top_n=" << metrics.rerank_top_n
              << " microbatch=" << metrics.microbatch
              << " topology_cache_ratio=" << metrics.topology_cache_ratio
              << " topology_cached_node_count="
              << metrics.topology_cached_node_count
              << " batches=" << metrics.batch_count
              << " first_batch=" << metrics.first_batch_queries
              << " last_batch=" << metrics.last_batch_queries
              << " tail_merged=" << (metrics.tail_batch_merged ? 1 : 0)
              << " lut_tile=" << metrics.lut_prefetch_tile_chunks
              << " num_queries=" << metrics.num_queries
              << " rvq_precompute_ms=" << metrics.rvq_precompute_ms
              << " pipeline_ms=" << metrics.pipeline_ms
              << " end_to_end_ms=" << metrics.end_to_end_ms
              << " pipeline_qps=" << metrics.pipeline_qps
              << " end_to_end_qps=" << metrics.end_to_end_qps
              << " recall@10=" << metrics.recall_at_10
              << " result_id_checksum=" << metrics.result_id_checksum
              << " avg_expanded=" << metrics.avg_expanded
              << " avg_visited=" << metrics.avg_visited
              << " avg_io_pages=" << metrics.avg_io_pages
              << " avg_rerank_ssd_io_pages=" << metrics.avg_rerank_ssd_io_pages
              << " avg_reused_exact=" << metrics.avg_reused_exact
              << " avg_exact_reuse_inserts=" << metrics.avg_exact_reuse_inserts
              << " avg_exact_reuse_overflows=" << metrics.avg_exact_reuse_overflows
              << " avg_topology_io_pages=" << metrics.avg_topology_io_pages
              << " avg_total_io_pages=" << metrics.avg_total_io_pages
              << " avg_exact=" << metrics.avg_io_pages
              << " stageA_ms=" << metrics.stage_a_wall_ms
              << " stageA_query_copy_ms=" << metrics.stage_a_query_copy_ms
              << " stageA_query_upload_ms=" << metrics.stage_a_query_upload_ms
              << " stageA_entry_batch_slice_ms=" << metrics.stage_a_entry_batch_slice_ms
              << " stageA_pq_query_tables_ms=" << metrics.stage_a_pq_query_tables_ms
              << " stageA_topology_stage_ms=" << metrics.stage_a_topology_stage_ms
              << " stageA_pq2_refine_stage_ms=" << metrics.stage_a_pq2_refine_stage_ms
              << " stageB_ms=" << metrics.stage_b_wall_ms
              << " topology_kernel_ms=" << metrics.topology_kernel_ms
              << " avg_query_pq_ms=" << metrics.avg_query_pq_ms
              << " avg_query_pq_compute_ms=" << metrics.avg_query_pq_compute_ms
              << " avg_query_pq_prefetch_issue_ms="
              << metrics.avg_query_pq_prefetch_issue_ms
              << " avg_query_pq_prefetch_wait_ms="
              << metrics.avg_query_pq_prefetch_wait_ms
              << " avg_query_queue_ms=" << metrics.avg_query_queue_ms
              << " pq_runtime_fraction=" << metrics.pq_runtime_fraction
              << " dynamic_shared_bytes=" << metrics.occupancy_dynamic_shared_bytes
              << " blocks_per_sm=" << metrics.occupancy_blocks_per_sm
              << " sm_count=" << metrics.occupancy_sm_count
              << " pq2_query_tables_ms=" << metrics.pq2_query_tables_ms
              << " pq2_refine_kernel_ms=" << metrics.pq2_refine_kernel_ms
              << " rerank_kernel_ms=" << metrics.rerank_kernel_ms
              << " topology_learned_stop_model_ms=" << metrics.topology_learned_stop_model_ms
              << " topology_learned_stop_feature_ms=" << metrics.topology_learned_stop_feature_ms
              << " topology_learned_stop_find_first_set_ms="
              << metrics.topology_learned_stop_find_first_set_ms
              << " topology_learned_stop_count_bits_ms="
              << metrics.topology_learned_stop_count_bits_ms
              << " topology_learned_stop_topk_churn_ms="
              << metrics.topology_learned_stop_topk_churn_ms
              << " topology_learned_stop_logit_eval_ms="
              << metrics.topology_learned_stop_logit_eval_ms
              << " learned_stop_model_ms=" << metrics.learned_stop_model_ms
              << " learned_stop_logit_eval_ms=" << metrics.learned_stop_logit_eval_ms
              << " rerank_query_block_ms=" << metrics.rerank_query_block_ms
              << " learned_stop_cycle_fraction="
              << (metrics.rerank_query_block_ms > 0.0
                      ? metrics.learned_stop_model_ms / metrics.rerank_query_block_ms
                      : 0.0)
              << " exact_reuse_lookup_ms=" << metrics.exact_reuse_lookup_ms
              << " combined_node_read_ms=" << metrics.combined_node_read_ms
              << " exact_reuse_insert_ms=" << metrics.exact_reuse_insert_ms
              << " exact_reuse_conservation_delta="
              << (metrics.avg_io_pages - metrics.avg_rerank_ssd_io_pages -
                  metrics.avg_reused_exact)
              << " gap=" << gap
              << " overlap_ratio=" << overlap_ratio
              << " rerank_bw_gbps=" << metrics.rerank_bw_gbps;
    if (std::string_view(mode) == "overlap") {
        std::cout << " overlap_efficiency=" << metrics.overlap_efficiency;
    }
    if (baseline.has_value()) {
        std::cout << " qps_ratio_vs_serial="
                  << (metrics.end_to_end_qps / baseline->end_to_end_qps)
                  << " topology_kernel_ratio_vs_serial="
                  << (metrics.topology_kernel_ms / baseline->topology_kernel_ms)
                  << " rerank_bw_ratio_vs_serial="
                  << (metrics.rerank_bw_gbps /
                      std::max(1e-9, baseline->rerank_bw_gbps));
    }
    std::cout << std::endl;
}

struct ScopedCudaStream {
    cudaStream_t stream = nullptr;

    ScopedCudaStream() {
        topoanns::ThrowIfCudaError(
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~ScopedCudaStream() {
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }

    ScopedCudaStream(const ScopedCudaStream&) = delete;
    ScopedCudaStream& operator=(const ScopedCudaStream&) = delete;
};

using BenchmarkClock = std::chrono::steady_clock;

struct TopologyLaunchGate {
    std::mutex mutex;
    std::condition_variable condition;
    bool launched = false;
    bool cancelled = false;
    bool io_submitted = false;
    bool io_failed = false;
    BenchmarkClock::time_point io_launch_time;
    BenchmarkClock::time_point launch_time;
};

void NotifyTopologyLaunch(void* context) {
    auto* gate = static_cast<TopologyLaunchGate*>(context);
    std::unique_lock<std::mutex> lock(gate->mutex);
    gate->launched = true;
    gate->condition.notify_all();
    gate->condition.wait(lock, [&] { return gate->io_submitted || gate->io_failed; });
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    gate->launch_time = BenchmarkClock::now();
}

void NotifyIoSubmitted(void* context) {
    auto* gate = static_cast<TopologyLaunchGate*>(context);
    {
        std::lock_guard<std::mutex> lock(gate->mutex);
        gate->io_launch_time = BenchmarkClock::now();
        gate->io_submitted = true;
    }
    gate->condition.notify_all();
}

struct IoOnlyMeasurement {
    topoanns::BamTraceBenchmarkResult run;
    std::uint64_t physical_reads = 0;
    double physical_iops = 0.0;
    double physical_gib_per_second = 0.0;
};

struct ConcurrentMeasurement {
    double topology_kernel_ms = 0.0;
    IoOnlyMeasurement io;
    double overlap_ms = 0.0;
    double topology_overlap_pct = 0.0;
    double io_overlap_pct = 0.0;
};

IoOnlyMeasurement RunIoOnlyMeasurement(
    topoanns::BamVectorPageProvider* owner,
    const topoanns::VectorPageProvider& provider,
    const topoanns::CudaBuffer<std::uint64_t>& device_trace,
    std::size_t start_index,
    std::size_t num_requests,
    std::size_t num_blocks,
    std::size_t warps_per_block,
    topoanns::BamTraceBenchmarkWorkspace* workspace,
    cudaStream_t stream) {
    owner->ResetIoDepthProfile(false);
    const topoanns::BamIoProfileSnapshot before = owner->ReadIoProfile();
    IoOnlyMeasurement measurement;
    measurement.run = topoanns::RunPreparedBam4kTraceBenchmark(
        provider, device_trace, start_index, num_requests, num_blocks,
        warps_per_block, workspace, stream, topoanns::kDefaultPageSizeBytes,
        topoanns::BamTraceRequestMapping::kThreadPerPage);
    const topoanns::BamIoProfileSnapshot after = owner->ReadIoProfile();
    measurement.physical_reads =
        CounterDelta(after.physical_reads, before.physical_reads);
    const double seconds = measurement.run.elapsed_ms / 1000.0;
    measurement.physical_iops =
        seconds == 0.0 ? 0.0 : measurement.physical_reads / seconds;
    measurement.physical_gib_per_second =
        seconds == 0.0
            ? 0.0
            : measurement.physical_reads *
                  static_cast<double>(topoanns::kDefaultPageSizeBytes) /
                  seconds / static_cast<double>(1ULL << 30);
    return measurement;
}

ConcurrentMeasurement RunConcurrentMeasurement(
    const topoanns::SearchResources& resources,
    const topoanns::PqDistanceOracle& oracle,
    const topoanns::DeviceEntryBatch& entry_batch,
    const topoanns::TopologySearchParams& topology_params,
    topoanns::BamVectorPageProvider* owner,
    const topoanns::VectorPageProvider& io_provider,
    const topoanns::CudaBuffer<std::uint64_t>& device_trace,
    std::size_t start_index,
    std::size_t num_requests,
    std::size_t num_blocks,
    std::size_t warps_per_block,
    topoanns::BamTraceBenchmarkWorkspace* workspace,
    std::uint32_t cuda_device,
    cudaStream_t topology_stream,
    cudaStream_t io_stream) {
    owner->ResetIoDepthProfile(false);
    const topoanns::BamIoProfileSnapshot before = owner->ReadIoProfile();
    TopologyLaunchGate gate;
    std::exception_ptr io_error;
    IoOnlyMeasurement io_measurement;

    std::thread io_thread([&] {
        try {
            topoanns::ThrowIfCudaError(cudaSetDevice(static_cast<int>(cuda_device)),
                                       "cudaSetDevice(topology I/O worker)");
            {
                std::unique_lock<std::mutex> lock(gate.mutex);
                gate.condition.wait(lock, [&] { return gate.launched || gate.cancelled; });
                if (gate.cancelled) {
                    return;
                }
            }
            io_measurement.run = topoanns::RunPreparedBam4kTraceBenchmark(
                io_provider, device_trace, start_index, num_requests, num_blocks,
                warps_per_block, workspace, io_stream,
                topoanns::kDefaultPageSizeBytes,
                topoanns::BamTraceRequestMapping::kThreadPerPage,
                NotifyIoSubmitted, &gate);
        } catch (...) {
            io_error = std::current_exception();
            {
                std::lock_guard<std::mutex> lock(gate.mutex);
                gate.io_failed = true;
            }
            gate.condition.notify_all();
        }
    });

    topoanns::detail::DeviceTopologyBatchResult topology_result;
    try {
        topology_result =
            topoanns::detail::RunTopologySearchKernelBatchDeviceWithLaunchCallback(
                resources, oracle, entry_batch, topology_params,
                NotifyTopologyLaunch, &gate, topology_stream);
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(gate.mutex);
            gate.cancelled = true;
        }
        gate.condition.notify_one();
        io_thread.join();
        throw;
    }
    io_thread.join();
    if (io_error != nullptr) {
        std::rethrow_exception(io_error);
    }

    const topoanns::BamIoProfileSnapshot after = owner->ReadIoProfile();
    io_measurement.physical_reads =
        CounterDelta(after.physical_reads, before.physical_reads);
    const double io_seconds = io_measurement.run.elapsed_ms / 1000.0;
    io_measurement.physical_iops =
        io_seconds == 0.0 ? 0.0 : io_measurement.physical_reads / io_seconds;
    io_measurement.physical_gib_per_second =
        io_seconds == 0.0
            ? 0.0
            : io_measurement.physical_reads *
                  static_cast<double>(topoanns::kDefaultPageSizeBytes) /
                  io_seconds / static_cast<double>(1ULL << 30);

    const auto topology_end =
        gate.launch_time + std::chrono::duration_cast<BenchmarkClock::duration>(
                               std::chrono::duration<double, std::milli>(
                                   topology_result.kernel_ms));
    const auto io_end =
        gate.io_launch_time + std::chrono::duration_cast<BenchmarkClock::duration>(
                       std::chrono::duration<double, std::milli>(
                           io_measurement.run.elapsed_ms));
    const auto overlap_begin = std::max(gate.launch_time, gate.io_launch_time);
    const auto overlap_end = std::min(topology_end, io_end);
    const double overlap_ms =
        overlap_end > overlap_begin
            ? std::chrono::duration<double, std::milli>(overlap_end - overlap_begin).count()
            : 0.0;

    ConcurrentMeasurement measurement;
    measurement.topology_kernel_ms = topology_result.kernel_ms;
    measurement.io = std::move(io_measurement);
    measurement.overlap_ms = overlap_ms;
    measurement.topology_overlap_pct =
        topology_result.kernel_ms == 0.0
            ? 0.0
            : 100.0 * overlap_ms / topology_result.kernel_ms;
    measurement.io_overlap_pct =
        measurement.io.run.elapsed_ms == 0.0
            ? 0.0
            : 100.0 * overlap_ms / measurement.io.run.elapsed_ms;
    return measurement;
}

double Mean(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) {
        sum += value;
    }
    return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
}

void RunTopologyIoConcurrencyBenchmark(
    const Args& args,
    const topoanns::BamRuntimeConfig& bam_runtime,
    const topoanns::BamVectorProviderOptions& bam_options,
    topoanns::SearchResources& resources,
    const FloatMatrix& queries,
    const topoanns::DeviceEntryBatch& full_entry_batch,
    std::uint64_t topology_cached_node_count,
    topoanns::BamVectorPageProvider* owner) {
    if (topology_cached_node_count != resources.num_nodes()) {
        throw std::runtime_error(
            "topology I/O concurrency benchmark requires topology-cache-ratio=1.0.");
    }
    const std::uint64_t payload_bytes =
        std::filesystem::file_size(args.combined_node_index) -
        topoanns::kDefaultPageSizeBytes;
    if (payload_bytes == 0 ||
        payload_bytes % topoanns::kDefaultPageSizeBytes != 0) {
        throw std::runtime_error("combined-node payload must be 4KB aligned.");
    }
    std::vector<std::uint64_t> trace =
        LoadPageTrace(args.topology_io_concurrency_trace);
    const std::size_t required_trace_entries =
        args.topology_io_warmup_requests + args.topology_io_measure_requests;
    if (trace.size() < required_trace_entries) {
        throw std::runtime_error("topology I/O concurrency trace is too short.");
    }
    const std::uint64_t payload_pages =
        payload_bytes / topoanns::kDefaultPageSizeBytes;
    const auto invalid = std::find_if(
        trace.begin(), trace.end(),
        [payload_pages](std::uint64_t page_id) { return page_id >= payload_pages; });
    if (invalid != trace.end()) {
        throw std::runtime_error("topology I/O concurrency trace exceeds payload range.");
    }

    topoanns::BamVectorProviderOptions io_options = bam_options;
    io_options.device_offset_bytes =
        *args.combined_node_ssd_device_offset_bytes;
    io_options.payload_bytes_override = payload_bytes;
    std::shared_ptr<topoanns::VectorPageProvider> io_provider =
        owner->CreateFileRangeProvider(
            args.combined_node_index, topoanns::kDefaultPageSizeBytes,
            topoanns::kDefaultPageSizeBytes, io_options);
    const topoanns::CudaBuffer<std::uint64_t> device_trace =
        topoanns::CudaBuffer<std::uint64_t>::CopyFromHost(trace);
    topoanns::BamTraceBenchmarkWorkspace io_workspace =
        topoanns::PrepareBamTraceBenchmarkWorkspace(args.topology_io_blocks);
    ScopedCudaStream topology_stream;
    ScopedCudaStream io_stream;

    topoanns::CudaBuffer<float> device_queries =
        topoanns::CudaBuffer<float>::CopyFromHost(queries.values);
    topoanns::PqQueryDistanceTables query_tables =
        topoanns::PqQueryDistanceTables::FromFloatQueriesDeviceBufferAsync(
            resources.pq_index(), device_queries, queries.rows,
            topology_stream.stream);
    topoanns::ThrowIfCudaError(cudaStreamSynchronize(topology_stream.stream),
                               "cudaStreamSynchronize(prepare topology workload)");
    const topoanns::PqDistanceOracle oracle(
        resources.pq_index(), std::move(query_tables));

    topoanns::TopologySearchParams topology_params;
    topology_params.top_k = args.top_k;
    topology_params.top_l = args.top_l_values.front();
    topology_params.candidate_queue_size = args.top_l_values.front();
    topology_params.search_width = args.search_width;
    topology_params.max_expansions = args.max_expansions;
    topology_params.topology_cached_node_count = topology_cached_node_count;

    const std::size_t warps_per_block = args.topology_io_threads / 32;
    std::cout << std::fixed << std::setprecision(3)
              << "[topology_io_concurrency_config]"
              << " topology_cache_ratio=1.000"
              << " queries=" << queries.rows
              << " top_l=" << args.top_l_values.front()
              << " io_blocks=" << args.topology_io_blocks
              << " io_threads=" << args.topology_io_threads
              << " io_warmup_requests=" << args.topology_io_warmup_requests
              << " io_measure_requests=" << args.topology_io_measure_requests
              << " measured_runs=" << args.measured_runs
              << std::endl;

    const std::size_t warmup_runs = std::max<std::size_t>(1, args.warmup_runs);
    for (std::size_t repeat = 0; repeat < warmup_runs; ++repeat) {
        topoanns::detail::DeviceTopologyBatchResult topology_warmup =
            topoanns::detail::RunTopologySearchKernelBatchDevice(
                resources, oracle, full_entry_batch, topology_params,
                topology_stream.stream);
        RunIoOnlyMeasurement(
            owner, *io_provider, device_trace, 0,
            args.topology_io_warmup_requests, args.topology_io_blocks,
            warps_per_block, &io_workspace, io_stream.stream);
        RunConcurrentMeasurement(
            resources, oracle, full_entry_batch, topology_params, owner,
            *io_provider, device_trace, 0, args.topology_io_warmup_requests,
            args.topology_io_blocks, warps_per_block, &io_workspace,
            bam_runtime.cuda_device, topology_stream.stream, io_stream.stream);
    }

    std::vector<double> topology_alone_ms;
    std::vector<double> topology_concurrent_ms;
    std::vector<double> io_alone_iops;
    std::vector<double> io_concurrent_iops;
    std::vector<double> io_alone_gib_s;
    std::vector<double> io_concurrent_gib_s;
    std::vector<double> topology_overlap_pct;
    std::vector<double> io_overlap_pct;
    for (std::size_t repeat = 1; repeat <= args.measured_runs; ++repeat) {
        const topoanns::detail::DeviceTopologyBatchResult topology_alone =
            topoanns::detail::RunTopologySearchKernelBatchDevice(
                resources, oracle, full_entry_batch, topology_params,
                topology_stream.stream);
        const IoOnlyMeasurement io_alone = RunIoOnlyMeasurement(
            owner, *io_provider, device_trace,
            args.topology_io_warmup_requests,
            args.topology_io_measure_requests, args.topology_io_blocks,
            warps_per_block, &io_workspace, io_stream.stream);
        const ConcurrentMeasurement concurrent = RunConcurrentMeasurement(
            resources, oracle, full_entry_batch, topology_params, owner,
            *io_provider, device_trace, args.topology_io_warmup_requests,
            args.topology_io_measure_requests, args.topology_io_blocks,
            warps_per_block, &io_workspace, bam_runtime.cuda_device,
            topology_stream.stream, io_stream.stream);

        topology_alone_ms.push_back(topology_alone.kernel_ms);
        topology_concurrent_ms.push_back(concurrent.topology_kernel_ms);
        io_alone_iops.push_back(io_alone.physical_iops);
        io_concurrent_iops.push_back(concurrent.io.physical_iops);
        io_alone_gib_s.push_back(io_alone.physical_gib_per_second);
        io_concurrent_gib_s.push_back(concurrent.io.physical_gib_per_second);
        topology_overlap_pct.push_back(concurrent.topology_overlap_pct);
        io_overlap_pct.push_back(concurrent.io_overlap_pct);

        std::cout << "[topology_io_concurrency] mode=topology-alone"
                  << " repeat=" << repeat
                  << " topology_kernel_ms=" << topology_alone.kernel_ms
                  << std::endl;
        std::cout << "[topology_io_concurrency] mode=io-alone"
                  << " repeat=" << repeat
                  << " io_kernel_ms=" << io_alone.run.elapsed_ms
                  << " physical_reads=" << io_alone.physical_reads
                  << " physical_iops=" << io_alone.physical_iops
                  << " physical_gib_s=" << io_alone.physical_gib_per_second
                  << std::endl;
        std::cout << "[topology_io_concurrency] mode=topology+io"
                  << " repeat=" << repeat
                  << " topology_kernel_ms=" << concurrent.topology_kernel_ms
                  << " io_kernel_ms=" << concurrent.io.run.elapsed_ms
                  << " physical_reads=" << concurrent.io.physical_reads
                  << " physical_iops=" << concurrent.io.physical_iops
                  << " physical_gib_s=" << concurrent.io.physical_gib_per_second
                  << " overlap_ms=" << concurrent.overlap_ms
                  << " topology_overlap_pct=" << concurrent.topology_overlap_pct
                  << " io_overlap_pct=" << concurrent.io_overlap_pct
                  << std::endl;
    }

    const double mean_topology_alone_ms = Mean(topology_alone_ms);
    const double mean_topology_concurrent_ms = Mean(topology_concurrent_ms);
    const double mean_io_alone_iops = Mean(io_alone_iops);
    const double mean_io_concurrent_iops = Mean(io_concurrent_iops);
    const double topology_slowdown_pct =
        mean_topology_alone_ms == 0.0
            ? 0.0
            : 100.0 * (mean_topology_concurrent_ms / mean_topology_alone_ms - 1.0);
    const double iops_loss_pct =
        mean_io_alone_iops == 0.0
            ? 0.0
            : 100.0 * (1.0 - mean_io_concurrent_iops / mean_io_alone_iops);
    std::cout << "[topology_io_concurrency_summary]"
              << " topology_alone_ms=" << mean_topology_alone_ms
              << " topology_concurrent_ms=" << mean_topology_concurrent_ms
              << " topology_slowdown_pct=" << topology_slowdown_pct
              << " io_alone_physical_iops=" << mean_io_alone_iops
              << " io_concurrent_physical_iops=" << mean_io_concurrent_iops
              << " iops_loss_pct=" << iops_loss_pct
              << " io_alone_physical_gib_s=" << Mean(io_alone_gib_s)
              << " io_concurrent_physical_gib_s=" << Mean(io_concurrent_gib_s)
              << " topology_overlap_pct=" << Mean(topology_overlap_pct)
              << " io_overlap_pct=" << Mean(io_overlap_pct)
              << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        g_bam_io_depth_profile = args.bam_io_depth_profile;
        g_topology_occupancy_profile = args.topology_occupancy_profile;
        g_topology_microbatch_io = args.topology_microbatch_io;
        g_topology_microbatch_queries = args.topology_microbatch_queries;
        g_topology_microbatch_contexts = args.topology_microbatch_contexts;
        g_topology_microbatch_io_blocks = args.topology_microbatch_io_blocks;
        g_topology_microbatch_io_threads = args.topology_microbatch_io_threads;
        const topoanns::BamRuntimeConfig bam_runtime = topoanns::ResolveBamRuntimeConfig(
            args.bam_config_path, args.bam_overrides, args.allow_bam_controller_override);
        topoanns::ThrowIfCudaError(cudaSetDevice(static_cast<int>(bam_runtime.cuda_device)),
                                   "cudaSetDevice");
        RequireL40Device(bam_runtime.cuda_device, "topoanns_overlap_pipeline_eval");
        ConfigureTraversalStop(args);

        const FloatMatrix queries = LoadFloatMatrix(args.query_bin);
        const IntMatrix gt = LoadIntMatrix(args.gt_bin);
        if (args.num_queries > queries.rows || args.num_queries > gt.rows) {
            throw std::runtime_error("num-queries exceeds query/gt rows.");
        }
        if (queries.cols == 0 || args.top_k > gt.cols) {
            throw std::runtime_error("Invalid query/GT dimensions.");
        }

        const std::filesystem::path topology_path = args.index_dir / "topology.bin";
        topoanns::SearchResources resources =
            topoanns::SearchResources::FromTopologyFile(topology_path);
        const std::filesystem::path base_pq_pivots =
            args.base_pq_pivots.empty() ? args.index_dir / "_pq_pivots.bin"
                                        : args.base_pq_pivots;
        const std::filesystem::path base_pq_codes =
            args.base_pq_codes.empty() ? args.index_dir / "_pq_compressed.bin"
                                       : args.base_pq_codes;
        resources.LoadPqIndex(base_pq_pivots, base_pq_codes);
        if (args.topology_io_concurrency_trace.empty() && !args.rerank_use_hpq &&
            !args.disable_pq2_refine) {
            resources.LoadPq2Index(args.pq2_pivots, args.pq2_codes, args.pq2_error_bounds);
        }
        if (args.rerank_use_hpq) {
            resources.LoadHpqIndex(args.hpq_base_pivots, args.hpq_outlier_pivots,
                                   args.hpq_codes,
                                   args.hpq_selector_bits);
        }
        resources.LoadVectorStore(args.index_dir / "vectors.ssd");

        topoanns::BamVectorProviderOptions bam_options;
        topoanns::ApplyBamRuntimeConfig(bam_runtime, &bam_options);
        topoanns::BamVectorProviderOptions vector_bam_options = bam_options;
        vector_bam_options.device_offset_bytes = args.bam_device_offset_bytes;
        topoanns::BamVectorProviderOptions owner_bam_options = bam_options;
        owner_bam_options.create_primary_range = false;
        auto bam_page_cache_owner = std::make_shared<topoanns::BamVectorPageProvider>(
            args.index_dir / "vectors.ssd",
            resources.vector_store_layout().header_bytes(),
            resources.vector_store_layout().page_size_bytes(),
            owner_bam_options);

        const std::vector<TopologyCacheSweepPoint> topology_cache_sweep =
            BuildTopologyCacheSweep(resources, args);
        const bool has_partial_topology = std::any_of(
            topology_cache_sweep.begin(), topology_cache_sweep.end(),
            [&](const TopologyCacheSweepPoint& point) {
                return point.cached_node_count < resources.num_nodes();
            });
        if (!args.topology_io_concurrency_trace.empty() &&
            topology_cache_sweep.size() != 1) {
            throw std::runtime_error(
                "topology I/O concurrency benchmark does not support a topology ratio sweep.");
        }
        ExactReuseControlConfig exact_reuse_control;
        exact_reuse_control.enabled = args.enable_exact_reuse;
        exact_reuse_control.query_dim = queries.cols;
        exact_reuse_control.cache_capacity = args.exact_reuse_cache_capacity;
        std::optional<topoanns::BamVectorProviderOptions> combined_bam_options;
        std::optional<topoanns::BamVectorProviderOptions> topology_bam_options;
        if (has_partial_topology && args.enable_exact_reuse) {
            if (args.combined_node_index.empty() ||
                !args.combined_node_ssd_device_offset_bytes.has_value()) {
                throw std::runtime_error(
                    "partial topology cache with exact reuse requires combined-node index and "
                    "SSD offset.");
            }
            const topoanns::DiskannDiskIndexLayout combined_layout =
                topoanns::DiskannDiskIndexLayout::Load(args.combined_node_index);
            if (combined_layout.metadata().num_nodes != resources.num_nodes() ||
                combined_layout.metadata().vector_dim != queries.cols ||
                combined_layout.metadata().nodes_per_sector == 0) {
                throw std::runtime_error("combined-node index metadata does not match search data.");
            }
            combined_bam_options = bam_options;
            combined_bam_options->device_offset_bytes =
                *args.combined_node_ssd_device_offset_bytes;
            combined_bam_options->payload_bytes_override =
                std::filesystem::file_size(args.combined_node_index) -
                topoanns::kDefaultPageSizeBytes;
            exact_reuse_control.node_bytes =
                static_cast<std::uint32_t>(combined_layout.metadata().max_node_len);
            exact_reuse_control.nodes_per_page =
                static_cast<std::uint32_t>(combined_layout.metadata().nodes_per_sector);
            std::cout << "[topoanns_combined_node_cache]"
                      << " sweep_points=" << topology_cache_sweep.size()
                      << " combined_node_ssd_device_offset_bytes="
                      << *args.combined_node_ssd_device_offset_bytes
                      << " node_bytes=" << exact_reuse_control.node_bytes
                      << " nodes_per_page=" << exact_reuse_control.nodes_per_page
                      << " cache_capacity=" << exact_reuse_control.cache_capacity
                      << std::endl;
            if (args.exact_reuse_ab_sequence) {
                if (!args.topology_ssd_device_offset_bytes.has_value()) {
                    throw std::runtime_error(
                        "exact reuse A/B sequence also requires topology-only SSD offset.");
                }
                const topoanns::TopologyHeader topology_header =
                    topoanns::TopologyLayout::ReadHeader(topology_path);
                topology_bam_options = bam_options;
                topology_bam_options->device_offset_bytes =
                    *args.topology_ssd_device_offset_bytes;
                topology_bam_options->payload_bytes_override =
                    AlignUp(topology_header.payload_bytes, topoanns::kDefaultPageSizeBytes);
            }
        } else if (has_partial_topology) {
            if (!args.topology_ssd_device_offset_bytes.has_value()) {
                throw std::runtime_error(
                    "topology cache is smaller than the full topology; pass "
                    "--topology-ssd-device-offset-bytes.");
            }
            const topoanns::TopologyHeader topology_header =
                topoanns::TopologyLayout::ReadHeader(topology_path);
            topology_bam_options = bam_options;
            topology_bam_options->device_offset_bytes =
                *args.topology_ssd_device_offset_bytes;
            topology_bam_options->payload_bytes_override =
                AlignUp(topology_header.payload_bytes, topoanns::kDefaultPageSizeBytes);
            std::cout << "[topoanns_topology_cache]"
                      << " sweep_points=" << topology_cache_sweep.size()
                      << " num_nodes=" << resources.num_nodes()
                      << " degree=" << resources.degree()
                      << " topology_ssd_device_offset_bytes="
                      << *args.topology_ssd_device_offset_bytes
                      << " topology_payload_bytes=" << topology_header.payload_bytes
                      << " topology_payload_bytes_padded="
                      << *topology_bam_options->payload_bytes_override
                      << std::endl;
        }
        std::vector<float> query_values(
            queries.values.begin(),
            queries.values.begin() + args.num_queries * static_cast<std::size_t>(queries.cols));
        topoanns::RvqEntryProfile rvq_profile;
        topoanns::DeviceEntryBatch full_entry_batch;
        {
            topoanns::RvqModel rvq_model = topoanns::RvqModel::Load(args.rvq_model);
            rvq_model.WarmUp();
            full_entry_batch = rvq_model.ComputeFloat32DeviceEntryBatch(
                query_values, args.num_queries, args.rvq_entry_count, 0, &rvq_profile);
        }
        const double rvq_precompute_ms = rvq_profile.total_ms;

        FloatMatrix query_subset;
        query_subset.rows = static_cast<std::uint32_t>(args.num_queries);
        query_subset.cols = queries.cols;
        query_subset.values = std::move(query_values);

        IntMatrix gt_subset;
        gt_subset.rows = static_cast<std::uint32_t>(args.num_queries);
        gt_subset.cols = gt.cols;
        gt_subset.values.assign(
            gt.values.begin(),
            gt.values.begin() + args.num_queries * static_cast<std::size_t>(gt.cols));

        const auto log_active_provider = [&](const char* active) {
            std::size_t free_bytes = 0;
            std::size_t total_bytes = 0;
            topoanns::ThrowIfCudaError(cudaMemGetInfo(&free_bytes, &total_bytes),
                                       "cudaMemGetInfo(exact_reuse_provider)");
            std::cout << "[topoanns_exact_reuse_provider] active=" << active
                      << " gpu_free_mib=" << (free_bytes / (1ULL << 20))
                      << " gpu_total_mib=" << (total_bytes / (1ULL << 20))
                      << std::endl;
        };
        bool has_active_data_provider = false;
        const auto clear_data_providers = [&] {
            topoanns::ThrowIfCudaError(cudaDeviceSynchronize(),
                                       "cudaDeviceSynchronize(before provider switch)");
            if (has_active_data_provider) {
                bam_page_cache_owner->FlushDevicePageCache();
                bam_page_cache_owner->ResetDevicePageCacheRanges();
            }
            resources.ClearCombinedNodePageProvider();
            resources.ClearTopologyPageProvider();
            resources.ClearVectorPageProvider();
            has_active_data_provider = false;
        };
        const auto activate_vector_provider = [&] {
            clear_data_providers();
            resources.AttachVectorPageProvider(
                bam_page_cache_owner->CreateFileRangeProvider(
                    args.index_dir / "vectors.ssd",
                    resources.vector_store_layout().header_bytes(),
                    resources.vector_store_layout().page_size_bytes(),
                    vector_bam_options));
            has_active_data_provider = true;
            log_active_provider("vector");
        };
        const auto activate_topology_provider = [&] {
            if (!topology_bam_options.has_value()) {
                throw std::runtime_error("topology provider options are unavailable.");
            }
            clear_data_providers();
            resources.AttachVectorPageProvider(
                bam_page_cache_owner->CreateFileRangeProvider(
                    args.index_dir / "vectors.ssd",
                    resources.vector_store_layout().header_bytes(),
                    resources.vector_store_layout().page_size_bytes(),
                    vector_bam_options));
            resources.AttachTopologyPageProvider(
                bam_page_cache_owner->CreateFileRangeProvider(
                    topology_path, sizeof(topoanns::TopologyHeader),
                    topoanns::kDefaultPageSizeBytes, *topology_bam_options));
            has_active_data_provider = true;
            log_active_provider("vector+topology");
        };
        const auto activate_combined_provider = [&] {
            if (!combined_bam_options.has_value()) {
                throw std::runtime_error("combined-node provider options are unavailable.");
            }
            clear_data_providers();
            std::shared_ptr<topoanns::VectorPageProvider> combined_provider =
                bam_page_cache_owner->CreateFileRangeProvider(
                    args.combined_node_index, topoanns::kDefaultPageSizeBytes,
                    topoanns::kDefaultPageSizeBytes, *combined_bam_options);
            resources.AttachVectorPageProvider(combined_provider);
            resources.AttachCombinedNodePageProvider(std::move(combined_provider));
            has_active_data_provider = true;
            log_active_provider("combined");
        };
        if (!args.topology_io_concurrency_trace.empty()) {
            std::cout << "[topoanns_exact_reuse_provider] active=none" << std::endl;
            if (!args.lut_prefetch_tile_sweep_values.empty()) {
                throw std::runtime_error(
                    "topology I/O concurrency benchmark does not support a tile sweep.");
            }
            std::size_t tile = args.lut_prefetch_tile_chunks;
            if (const auto it = args.lut_prefetch_tile_by_top_l.find(args.top_l_values.front());
                it != args.lut_prefetch_tile_by_top_l.end()) {
                tile = it->second;
            }
            ConfigureFrontierPqMapping(args.frontier_pq_mapping, tile);
            RunTopologyIoConcurrencyBenchmark(
                args, bam_runtime, bam_options, resources, query_subset,
                full_entry_batch, topology_cache_sweep.front().cached_node_count,
                bam_page_cache_owner.get());
            return 0;
        }
        const RerankControlConfig rerank_control{
        args.rerank_use_early_stop,
        args.rerank_use_learned_stop,
        args.rerank_use_hpq,
        !args.disable_pq2_refine,
        args.rerank_use_pq2_bound_filter,
        args.rerank_early_stop_min_prefix,
            args.rerank_early_stop_patience_tiles,
            args.rerank_rank_tile_size,
            args.rerank_learned_stop_model,
        };

        std::vector<std::size_t> topology_execution_sweep =
            args.topology_microbatch_query_values;
        if (topology_execution_sweep.empty()) {
            topology_execution_sweep.push_back(
                args.topology_microbatch_io ? args.topology_microbatch_queries : 0U);
        }

        std::vector<std::size_t> topology_io_block_sweep =
            args.topology_microbatch_io_block_values;
        if (topology_io_block_sweep.empty()) {
            topology_io_block_sweep.push_back(
                args.topology_microbatch_io_blocks);
        }

        std::vector<std::pair<std::size_t, std::size_t>>
            topology_execution_configs;
        for (const std::size_t execution_queries : topology_execution_sweep) {
            if (execution_queries == 0U) {
                topology_execution_configs.emplace_back(
                    execution_queries, args.topology_microbatch_io_blocks);
                continue;
            }
            for (const std::size_t io_blocks : topology_io_block_sweep) {
                topology_execution_configs.emplace_back(
                    execution_queries, io_blocks);
            }
        }

        std::vector<std::optional<std::size_t>> tile_sweep;
        if (args.lut_prefetch_tile_sweep_values.empty()) {
            tile_sweep.push_back(std::nullopt);
        } else {
            for (const std::size_t tile : args.lut_prefetch_tile_sweep_values) {
                tile_sweep.push_back(tile);
            }
        }

        if (args.exact_reuse_ab_sequence &&
            (topology_cache_sweep.size() != 1 || tile_sweep.size() != 1 ||
             args.top_l_values.size() != 1 || args.microbatch_values.size() != 1)) {
            throw std::runtime_error(
                "exact reuse A/B sequence requires one topology ratio, tile, top-L, and "
                "microbatch.");
        }

        for (const TopologyCacheSweepPoint& topology_cache_point :
             topology_cache_sweep) {
            const double topology_cache_ratio = topology_cache_point.ratio;
            const std::uint64_t topology_cached_node_count =
                topology_cache_point.cached_node_count;
            const double topology_cache_gib =
                static_cast<double>(topology_cached_node_count) *
                static_cast<double>(resources.degree()) * sizeof(std::uint32_t) /
                static_cast<double>(1ULL << 30);
            std::cout << "[topoanns_topology_cache_sweep]"
                      << " topology_cache_ratio=" << topology_cache_ratio
                      << " topology_cached_node_count=" << topology_cached_node_count
                      << " topology_cache_gib=" << topology_cache_gib
                      << " topology_loaded_once=1"
                      << std::endl;
            if (topology_cached_node_count == resources.num_nodes()) {
                activate_vector_provider();
            } else if (args.enable_exact_reuse) {
                if (!args.exact_reuse_ab_sequence) {
                    activate_combined_provider();
                }
            } else {
                activate_topology_provider();
            }

            for (const std::optional<std::size_t> tile_override : tile_sweep) {
            for (std::size_t sweep_index = 0; sweep_index < args.top_l_values.size(); ++sweep_index) {
                const std::size_t top_l = args.top_l_values[sweep_index];
                const std::size_t rerank_top_n = args.rerank_top_n_values[sweep_index];
                std::size_t lut_prefetch_tile_chunks = args.lut_prefetch_tile_chunks;
                if (tile_override.has_value()) {
                    lut_prefetch_tile_chunks = *tile_override;
                } else if (const auto it = args.lut_prefetch_tile_by_top_l.find(top_l);
                           it != args.lut_prefetch_tile_by_top_l.end()) {
                    lut_prefetch_tile_chunks = it->second;
                }
                ConfigureFrontierPqMapping(args.frontier_pq_mapping, lut_prefetch_tile_chunks);
                for (const std::size_t microbatch : args.microbatch_values) {
                    const auto run_one = [&](RunKind run_kind,
                                             std::size_t repeat_id,
                                             const ExactReuseControlConfig& active_exact_reuse,
                                             const char* exact_reuse_variant) {
                        std::optional<ExperimentMetrics> serial;
                        if (args.pipeline_mode == PipelineMode::kSerial ||
                            args.pipeline_mode == PipelineMode::kBoth) {
                            const std::size_t serial_microbatch =
                                args.serial_microbatch == 0 ? microbatch
                                                            : args.serial_microbatch;
                            serial = RunSerialExperiment(
                                resources, query_subset, gt_subset, full_entry_batch,
                                top_l, rerank_top_n, serial_microbatch,
                                args.top_k, args.search_width, args.max_expansions,
                                topology_cached_node_count, active_exact_reuse, rerank_control,
                                rvq_precompute_ms, bam_runtime.cuda_device,
                                args.merge_small_tail_batch,
                                args.bam_io_depth_profile ? bam_page_cache_owner.get() : nullptr,
                                run_kind == RunKind::kMeasured
                                    ? args.expanded_trace_output_dir : std::filesystem::path{});
                            serial->lut_prefetch_tile_chunks = lut_prefetch_tile_chunks;
                            serial->topology_cache_ratio = topology_cache_ratio;
                            serial->topology_cached_node_count =
                                topology_cached_node_count;
                            PrintMetrics("serial", exact_reuse_variant, *serial, std::nullopt,
                                         run_kind, repeat_id);
                        }
                        if (args.pipeline_mode == PipelineMode::kOverlap ||
                            args.pipeline_mode == PipelineMode::kBoth) {
                            ExperimentMetrics overlap = RunOverlapExperiment(
                                resources, query_subset, gt_subset, full_entry_batch,
                                top_l, rerank_top_n, microbatch,
                                args.top_k, args.search_width, args.max_expansions,
                                topology_cached_node_count, active_exact_reuse, rerank_control,
                                rvq_precompute_ms, bam_runtime.cuda_device, args.ring_depth,
                                args.merge_small_tail_batch);
                            overlap.lut_prefetch_tile_chunks = lut_prefetch_tile_chunks;
                            overlap.topology_cache_ratio = topology_cache_ratio;
                            overlap.topology_cached_node_count =
                                topology_cached_node_count;
                            PrintMetrics("overlap", exact_reuse_variant, overlap, serial,
                                         run_kind, repeat_id);
                        }
                    };
                    const auto run_repeats =
                        [&](const ExactReuseControlConfig& control, const char* variant) {
                            for (std::size_t repeat_id = 1;
                                 repeat_id <= args.warmup_runs; ++repeat_id) {
                                run_one(RunKind::kWarmup, repeat_id, control, variant);
                            }
                            for (std::size_t repeat_id = 1;
                                 repeat_id <= args.measured_runs; ++repeat_id) {
                                run_one(RunKind::kMeasured, repeat_id, control, variant);
                            }
                        };
                    for (const auto& [execution_queries, io_blocks] :
                         topology_execution_configs) {
                        g_topology_microbatch_io = execution_queries != 0U;
                        if (g_topology_microbatch_io) {
                            g_topology_microbatch_queries = execution_queries;
                            g_topology_microbatch_io_blocks = io_blocks;
                        }
                        std::cout << "[topoanns_topology_execution_config]"
                                  << " microbatch_io="
                                  << (g_topology_microbatch_io ? 1 : 0)
                                  << " microbatch_queries=" << execution_queries
                                  << " context_depth="
                                  << g_topology_microbatch_contexts
                                  << " io_blocks="
                                  << g_topology_microbatch_io_blocks
                                  << " io_threads="
                                  << g_topology_microbatch_io_threads
                                  << std::endl;
                        if (!args.exact_reuse_ab_sequence) {
                            run_repeats(
                                exact_reuse_control,
                                exact_reuse_control.enabled ? "reuse" : "old");
                        } else {
                            ExactReuseControlConfig old_control =
                                exact_reuse_control;
                            old_control.enabled = false;
                            if (topology_cached_node_count <
                                resources.num_nodes()) {
                                activate_topology_provider();
                            }
                            run_repeats(old_control, "old_before");
                            if (topology_cached_node_count <
                                resources.num_nodes()) {
                                activate_combined_provider();
                            }
                            run_repeats(exact_reuse_control, "reuse");
                            if (topology_cached_node_count <
                                resources.num_nodes()) {
                                activate_topology_provider();
                            }
                            run_repeats(old_control, "old_after");
                        }
                    }
                }
            }
        }
        }
        topoanns::detail::PrintRerankReuseProfileSummary();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_overlap_pipeline_eval] " << e.what() << std::endl;
        return 1;
    }
}
