#include "topoanns/bam_runtime_config.hpp"
#include "topoanns/bam_vector_provider.hpp"
#include "topoanns/rvq_entry_provider.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/topoanns_search.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::filesystem::path index_dir;
    std::filesystem::path rvq_model;
    std::filesystem::path query_bin;
    std::filesystem::path gt_bin;
    std::filesystem::path pq2_pivots;
    std::filesystem::path pq2_codes;
    std::filesystem::path pq2_error_bounds;
    std::filesystem::path rerank_learned_stop_model;
    std::size_t num_queries = 0;
    std::size_t batch_size = 256;
    std::size_t top_k = 10;
    std::vector<std::size_t> top_l_values;
    std::size_t search_width = 2;
    std::size_t rerank_top_n = 32;
    std::vector<std::size_t> rerank_top_n_values;
    std::vector<std::size_t> rerank_top_n_sweep_values;
    topoanns::RerankExecutionMode rerank_mode = topoanns::RerankExecutionMode::kPersistent;
    std::vector<topoanns::RerankExecutionMode> rerank_mode_sweep_values;
    std::size_t rerank_rank_tile_size = 16;
    std::size_t max_expansions = 4096;
    std::size_t rvq_entry_count = 128;
    std::vector<std::size_t> eval_query_counts;
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
    bool has_gt = false;
    std::vector<std::string> frontier_pq_modes = {"current"};
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

struct EvalProfileTotals {
    double rvq_entry_ms = 0.0;
    double rvq_kernel_ms = 0.0;
    double rvq_entry_gather_ms = 0.0;
    double rvq_host_expand_ms = 0.0;
    double query_h2d_ms = 0.0;
    double rvq_query_upload_ms = 0.0;
    double pq_query_upload_ms = 0.0;
    double rerank_query_upload_ms = 0.0;
    double pq_zero_fill_ms = 0.0;
    double pq_table_kernel_ms = 0.0;
    double pq_table_download_ms = 0.0;
    double topology_kernel_ms = 0.0;
    double topology_candidate_download_ms = 0.0;
    double topology_stats_download_ms = 0.0;
    double topology_host_postprocess_ms = 0.0;
    double topology_candidate_materialize_ms = 0.0;
    double topology_topk_extract_ms = 0.0;
    double topology_sum_query_pq_ms = 0.0;
    double topology_sum_query_pq_compute_ms = 0.0;
    double topology_sum_query_pq_prefetch_issue_ms = 0.0;
    double topology_sum_query_pq_prefetch_wait_ms = 0.0;
    double topology_sum_query_pq_checksum_ms = 0.0;
    double topology_sum_query_queue_ms = 0.0;
    double topology_sum_query_queue_scan_ms = 0.0;
    double topology_sum_query_queue_select_ms = 0.0;
    double topology_sum_query_frontier_sort_ms = 0.0;
    double topology_sum_query_tail_merge_ms = 0.0;
    double topology_sum_query_candidate_sort_ms = 0.0;
    double topology_sum_query_hash_rebuild_ms = 0.0;
    double rerank_prepare_ms = 0.0;
    double rerank_pq2_query_tables_ms = 0.0;
    double rerank_pq2_query_upload_ms = 0.0;
    double rerank_pq2_query_zero_fill_ms = 0.0;
    double rerank_pq2_query_table_kernel_ms = 0.0;
    double rerank_pq2_query_table_download_ms = 0.0;
    double rerank_pq2_refine_kernel_ms = 0.0;
    double rerank_pq2_bound_filter_ms = 0.0;
    double rerank_pq2_threshold_update_ms = 0.0;
    double rerank_learned_model_ms = 0.0;
    double rerank_learned_checkpoint_bookkeeping_ms = 0.0;
    double rerank_learned_topk_churn_ms = 0.0;
    double rerank_learned_next_window_scan_ms = 0.0;
    double rerank_learned_logit_eval_ms = 0.0;
    double rerank_io_ms = 0.0;
    double rerank_unpack_ms = 0.0;
    double rerank_exact_kernel_ms = 0.0;
    double rerank_sort_ms = 0.0;
    double rerank_result_download_ms = 0.0;
    double rerank_topk_extract_ms = 0.0;
    double rerank_sorted_candidates_materialize_ms = 0.0;
    std::size_t rerank_batches = 0;
    std::size_t pq2_refine_batches = 0;
    std::size_t pq2_residual_refine_batches = 0;
    std::size_t rerank_bam_direct_batches = 0;
    std::size_t rerank_gpu_sort_batches = 0;
    std::size_t rerank_host_sort_batches = 0;
    std::size_t rerank_requested_top_n = 0;
    std::size_t rerank_effective_top_n = 0;
    std::size_t pq2_refine_top_l = 0;
    std::size_t pq2_refine_input_candidates = 0;
    std::size_t pq2_refine_input_candidates_min = 0;
    std::size_t pq2_refine_input_candidates_max = 0;
    std::size_t rerank_tile_size = 0;
    std::size_t rerank_prepare_valid_candidates = 0;
    std::size_t rerank_prepare_valid_candidates_min = 0;
    std::size_t rerank_prepare_valid_candidates_max = 0;
    std::size_t rerank_prepare_filtered_candidates = 0;
    std::size_t rerank_prepare_filtered_candidates_min = 0;
    std::size_t rerank_prepare_filtered_candidates_max = 0;
    std::size_t rerank_prepare_nonzero_tiles = 0;
    std::size_t rerank_prepare_zero_tiles = 0;
    std::size_t rerank_learned_checkpoints = 0;
    std::size_t rerank_learned_stop_queries = 0;
    std::size_t rerank_learned_stop_prefix_sum = 0;
    std::size_t rerank_learned_stop_prefix_min = 0;
    std::size_t rerank_learned_stop_prefix_max = 0;
};

struct WarmQpsBreakdown {
    double query_h2d_ms = 0.0;
    double rvq_entry_compute_ms = 0.0;
    double pq_lut_compute_ms = 0.0;
    double topology_search_ms = 0.0;
    double rerank_candidate_organization_ms = 0.0;
    double page_read_ms = 0.0;
    double vector_unpack_ms = 0.0;
    double exact_distance_ms = 0.0;
    double final_topk_select_ms = 0.0;
    double final_topk_d2h_ms = 0.0;

    double total_ms() const {
        return query_h2d_ms + rvq_entry_compute_ms + pq_lut_compute_ms + topology_search_ms +
               rerank_candidate_organization_ms + page_read_ms + vector_unpack_ms +
               exact_distance_ms + final_topk_select_ms + final_topk_d2h_ms;
    }
};

struct InitProfile {
    double query_load_ms = 0.0;
    double gt_load_ms = 0.0;
    double topology_load_ms = 0.0;
    double pq_load_ms = 0.0;
    double vector_store_header_ms = 0.0;
    double bam_provider_init_ms = 0.0;
    double rvq_load_ms = 0.0;
};

void MergeMinMaxTotals(std::size_t src_min,
                       std::size_t src_max,
                       bool has_src,
                       std::size_t* dst_min,
                       std::size_t* dst_max,
                       bool* has_dst) {
    if (!has_src) {
        return;
    }
    if (!*has_dst) {
        *dst_min = src_min;
        *dst_max = src_max;
        *has_dst = true;
        return;
    }
    *dst_min = std::min(*dst_min, src_min);
    *dst_max = std::max(*dst_max, src_max);
}

struct EvalAggregate {
    std::size_t total_io_pages = 0;
    std::size_t total_exact_count = 0;
    std::size_t total_bound_filtered = 0;
    std::size_t total_visited = 0;
    std::size_t total_expanded = 0;
    double recall_sum = 0.0;
    EvalProfileTotals profile_totals;
};

struct SweepPoint {
    std::string frontier_pq_mode;
    std::size_t top_l = 0;
    std::size_t rerank_top_n = 0;
    topoanns::RerankExecutionMode rerank_mode = topoanns::RerankExecutionMode::kPersistent;
    std::size_t num_queries = 0;
    double qps = 0.0;
    double wall_qps = 0.0;
    double recall_at_10 = -1.0;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_eval_sift"
        << " --index-dir <path>"
        << " --rvq-model <path>"
        << " --query-bin <path>"
        << " [--gt-bin <path>]"
        << " --num-queries <count>"
        << " [--eval-query-counts <csv>]"
        << " --top-l-values <csv>"
        << " [--batch-size <count>]"
        << " [--top-k <count>]"
        << " [--search-width <count>]"
        << " [--rerank-top-n <count>]"
        << " [--rerank-top-n-values <csv>]"
        << " [--rerank-top-n-sweep-values <csv>]"
        << " [--rerank-mode <persistent>]"
        << " [--rerank-mode-sweep <csv>]"
        << " [--rerank-rank-tile-size <count>]"
        << " [--pq2-pivots <path> --pq2-codes <path> [--pq2-error-bounds <path>] --rerank-use-pq2]"
        << " [--rerank-use-pq2-bound-filter]"
        << " [--rerank-use-early-stop]"
        << " [--rerank-use-learned-stop --rerank-learned-stop-model <path>]"
        << " [--rerank-early-stop-min-prefix <count>]"
        << " [--rerank-early-stop-patience-tiles <count>]"
        << " [--frontier-pq-modes <csv>]"
        << " [--max-expansions <count>]"
        << " [--rvq-entry-count <count>]"
        << " [--bam-config-path <path>]"
        << " [--allow-bam-controller-override]"
        << " [--bam-controller-path <path>]"
        << " [--bam-device-offset-bytes <bytes>]"
        << " [--bam-page-cache-bytes <bytes>]"
        << " [--bam-queue-depth <count>]"
        << " [--bam-num-queues <count>]"
        << " [--bam-cuda-device <id>]"
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
        throw std::runtime_error("top-l-values must not be empty.");
    }
    return values;
}

std::vector<std::string> ParseCsvStrings(const std::string& text) {
    std::vector<std::string> values;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token =
            text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            values.push_back(token);
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return values;
}

void ConfigureFrontierPqMode(const std::string& mode) {
    if (mode == "current") {
        setenv("TOPOANNS_FRONTIER_PQ_WARPS", "4", 1);
        unsetenv("TOPOANNS_FRONTIER_PQ_MODE");
        return;
    }
    unsetenv("TOPOANNS_FRONTIER_PQ_WARPS");
    setenv("TOPOANNS_FRONTIER_PQ_MODE", mode.c_str(), 1);
}

topoanns::RerankExecutionMode ParseRerankMode(const std::string& text) {
    if (text == "persistent") {
        return topoanns::RerankExecutionMode::kPersistent;
    }
    throw std::runtime_error(
        "Unsupported rerank mode: " + text +
        ". Main evaluation now requires pure-GPU persistent rerank.");
}

const char* RerankModeName(topoanns::RerankExecutionMode mode) {
    switch (mode) {
        case topoanns::RerankExecutionMode::kLinear:
            return "linear";
        case topoanns::RerankExecutionMode::kTiled:
            return "tiled";
        case topoanns::RerankExecutionMode::kPageByPage:
            return "page";
        case topoanns::RerankExecutionMode::kPersistent:
            return "persistent";
    }
    return "unknown";
}

void RequireL40Device(std::uint32_t device_id, const char* context) {
    cudaDeviceProp props{};
    const cudaError_t status =
        cudaGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) +
                                 ": cudaGetDeviceProperties failed for requested BAM GPU.");
    }
    const std::string gpu_name(props.name);
    if (gpu_name.find("L40") == std::string::npos) {
        throw std::runtime_error(std::string(context) +
                                 ": BAM path must run on an NVIDIA L40, but the selected CUDA "
                                 "device is \"" + gpu_name + "\" (logical cuda:" +
                                 std::to_string(device_id) + ").");
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
            args.has_gt = true;
        } else if (flag == "--pq2-pivots") {
            args.pq2_pivots = read_value("--pq2-pivots");
        } else if (flag == "--pq2-codes") {
            args.pq2_codes = read_value("--pq2-codes");
        } else if (flag == "--pq2-error-bounds") {
            args.pq2_error_bounds = read_value("--pq2-error-bounds");
        } else if (flag == "--rerank-learned-stop-model") {
            args.rerank_learned_stop_model = read_value("--rerank-learned-stop-model");
        } else if (flag == "--num-queries") {
            args.num_queries = std::stoull(read_value("--num-queries"));
        } else if (flag == "--eval-query-counts") {
            args.eval_query_counts = ParseCsvSizes(read_value("--eval-query-counts"));
        } else if (flag == "--batch-size") {
            args.batch_size = std::stoull(read_value("--batch-size"));
        } else if (flag == "--top-k") {
            args.top_k = std::stoull(read_value("--top-k"));
        } else if (flag == "--top-l-values") {
            args.top_l_values = ParseCsvSizes(read_value("--top-l-values"));
        } else if (flag == "--search-width") {
            args.search_width = std::stoull(read_value("--search-width"));
        } else if (flag == "--rerank-top-n") {
            args.rerank_top_n = std::stoull(read_value("--rerank-top-n"));
        } else if (flag == "--rerank-top-n-values") {
            args.rerank_top_n_values = ParseCsvSizes(read_value("--rerank-top-n-values"));
        } else if (flag == "--rerank-top-n-sweep-values") {
            args.rerank_top_n_sweep_values =
                ParseCsvSizes(read_value("--rerank-top-n-sweep-values"));
        } else if (flag == "--rerank-mode") {
            args.rerank_mode = ParseRerankMode(read_value("--rerank-mode"));
        } else if (flag == "--rerank-mode-sweep") {
            for (const auto& token : ParseCsvStrings(read_value("--rerank-mode-sweep"))) {
                args.rerank_mode_sweep_values.push_back(ParseRerankMode(token));
            }
        } else if (flag == "--rerank-rank-tile-size") {
            args.rerank_rank_tile_size = std::stoull(read_value("--rerank-rank-tile-size"));
        } else if (flag == "--rerank-use-pq2") {
            args.use_pq2_refine = true;
        } else if (flag == "--rerank-use-pq2-bound-filter") {
            args.use_pq2_bound_filter = true;
        } else if (flag == "--rerank-use-early-stop") {
            args.use_early_stop = true;
        } else if (flag == "--rerank-use-learned-stop") {
            args.use_learned_stop = true;
        } else if (flag == "--rerank-early-stop-min-prefix") {
            args.rerank_early_stop_min_prefix =
                std::stoull(read_value("--rerank-early-stop-min-prefix"));
        } else if (flag == "--rerank-early-stop-patience-tiles") {
            args.rerank_early_stop_patience_tiles =
                std::stoull(read_value("--rerank-early-stop-patience-tiles"));
        } else if (flag == "--frontier-pq-modes") {
            args.frontier_pq_modes = ParseCsvStrings(read_value("--frontier-pq-modes"));
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
            args.bam_device_offset_bytes =
                std::stoull(read_value("--bam-device-offset-bytes"));
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
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }

    if (args.index_dir.empty() || args.rvq_model.empty() || args.query_bin.empty() ||
        args.num_queries == 0 || args.top_l_values.empty()) {
        Usage();
    }
    if (args.frontier_pq_modes.empty()) {
        throw std::runtime_error("frontier-pq-modes must not be empty.");
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
    if (args.eval_query_counts.empty()) {
        args.eval_query_counts.push_back(args.num_queries);
    }
    std::sort(args.eval_query_counts.begin(), args.eval_query_counts.end());
    args.eval_query_counts.erase(
        std::unique(args.eval_query_counts.begin(), args.eval_query_counts.end()),
        args.eval_query_counts.end());
    if (args.eval_query_counts.back() > args.num_queries) {
        throw std::runtime_error("eval-query-counts must be <= num-queries.");
    }
    if (!args.rerank_top_n_values.empty() &&
        args.rerank_top_n_values.size() != args.top_l_values.size()) {
        throw std::runtime_error("rerank-top-n-values must match top-l-values in length.");
    }
    if (!args.rerank_top_n_values.empty() && !args.rerank_top_n_sweep_values.empty()) {
        throw std::runtime_error(
            "rerank-top-n-values and rerank-top-n-sweep-values are mutually exclusive.");
    }
    if (!args.rerank_mode_sweep_values.empty()) {
        std::sort(args.rerank_mode_sweep_values.begin(), args.rerank_mode_sweep_values.end(),
                  [](auto lhs, auto rhs) {
                      return static_cast<int>(lhs) < static_cast<int>(rhs);
                  });
        args.rerank_mode_sweep_values.erase(
            std::unique(args.rerank_mode_sweep_values.begin(),
                        args.rerank_mode_sweep_values.end()),
            args.rerank_mode_sweep_values.end());
    }
    if (args.rerank_mode != topoanns::RerankExecutionMode::kPersistent) {
        throw std::runtime_error(
            "topoanns_eval_sift now requires pure-GPU persistent rerank.");
    }
    for (const auto mode : args.rerank_mode_sweep_values) {
        if (mode != topoanns::RerankExecutionMode::kPersistent) {
            throw std::runtime_error(
                "rerank-mode-sweep includes a non-persistent mode; pure-GPU runs only support "
                "persistent.");
        }
    }
    return args;
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

double RecallAtK(const topoanns::TopoAnnsBatchResult& result,
                 const IntMatrix& gt,
                 std::size_t gt_offset,
                 std::size_t top_k) {
    std::size_t matched = 0;
    for (std::size_t query_id = 0; query_id < result.queries.size(); ++query_id) {
        const auto& found = result.queries[query_id].rerank.topk;
        const std::int32_t* gt_row =
            gt.values.data() + (gt_offset + query_id) * static_cast<std::size_t>(gt.cols);
        for (const auto& candidate : found) {
            for (std::size_t rank = 0; rank < top_k; ++rank) {
                if (static_cast<std::int32_t>(candidate.node_id) == gt_row[rank]) {
                    ++matched;
                    break;
                }
            }
        }
    }
    return static_cast<double>(matched) /
           static_cast<double>(result.queries.size() * top_k);
}

WarmQpsBreakdown BuildWarmQpsBreakdown(const EvalProfileTotals& totals) {
    WarmQpsBreakdown breakdown;
    breakdown.query_h2d_ms = totals.rvq_query_upload_ms;
    breakdown.rvq_entry_compute_ms = totals.rvq_kernel_ms + totals.rvq_entry_gather_ms;
    breakdown.pq_lut_compute_ms = totals.pq_zero_fill_ms + totals.pq_table_kernel_ms;
    breakdown.topology_search_ms = totals.topology_kernel_ms;
    breakdown.rerank_candidate_organization_ms =
        totals.topology_candidate_download_ms + totals.topology_candidate_materialize_ms +
        totals.rerank_prepare_ms + totals.rerank_pq2_threshold_update_ms;
    breakdown.page_read_ms = totals.rerank_io_ms;
    breakdown.vector_unpack_ms = totals.rerank_unpack_ms;
    breakdown.exact_distance_ms = totals.rerank_exact_kernel_ms;
    breakdown.final_topk_select_ms = totals.rerank_sort_ms + totals.rerank_topk_extract_ms;
    breakdown.final_topk_d2h_ms = totals.rerank_result_download_ms;
    return breakdown;
}

double QpsFromMs(std::size_t num_queries, double total_ms) {
    return total_ms > 0.0 ? static_cast<double>(num_queries) / (total_ms / 1000.0) : 0.0;
}

void PrintSummaryLine(const std::string& frontier_pq_mode,
                      std::size_t top_l,
                      std::size_t rerank_top_n,
                      topoanns::RerankExecutionMode rerank_mode,
                      std::size_t num_queries,
                      double wall_elapsed_sec,
                      const WarmQpsBreakdown& warm_breakdown,
                      double recall_at_10,
                      std::size_t io_pages,
                      std::size_t exact_count,
                      std::size_t bound_filtered_count,
                      std::size_t visited_nodes,
                      std::size_t expanded_nodes) {
    const double warm_elapsed_sec = warm_breakdown.total_ms() / 1000.0;
    const double qps = QpsFromMs(num_queries, warm_breakdown.total_ms());
    const double wall_qps =
        wall_elapsed_sec > 0.0 ? static_cast<double>(num_queries) / wall_elapsed_sec : 0.0;
    const double avg_io_pages =
        num_queries == 0 ? 0.0 : static_cast<double>(io_pages) / num_queries;
    const double avg_exact =
        num_queries == 0 ? 0.0 : static_cast<double>(exact_count) / num_queries;
    const double avg_bound_filtered =
        num_queries == 0 ? 0.0 : static_cast<double>(bound_filtered_count) / num_queries;
    const double avg_visited =
        num_queries == 0 ? 0.0 : static_cast<double>(visited_nodes) / num_queries;
    const double avg_expanded =
        num_queries == 0 ? 0.0 : static_cast<double>(expanded_nodes) / num_queries;
    std::cout << "[topoanns_eval] frontier_pq_mode=" << frontier_pq_mode
              << " top_l=" << top_l
              << " rerank_top_n=" << rerank_top_n
              << " rerank_mode=" << RerankModeName(rerank_mode)
              << " queries=" << num_queries
              << " elapsed_sec=" << warm_elapsed_sec
              << " wall_elapsed_sec=" << wall_elapsed_sec
              << " qps=" << qps
              << " wall_qps=" << wall_qps;
    if (recall_at_10 >= 0.0) {
        std::cout << " recall@10=" << recall_at_10;
    }
    std::cout << " avg_io_pages=" << avg_io_pages
              << " avg_exact=" << avg_exact
              << " avg_bound_filtered=" << avg_bound_filtered
              << " avg_visited=" << avg_visited
              << " avg_expanded=" << avg_expanded
              << std::endl;
}

void PrintProfileLine(const std::string& frontier_pq_mode,
                      std::size_t top_l,
                      std::size_t rerank_top_n,
                      topoanns::RerankExecutionMode rerank_mode,
                      std::size_t num_queries,
                      const EvalProfileTotals& totals) {
    const double avg_query_pq_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_pq_ms / num_queries;
    const double avg_query_pq_compute_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_pq_compute_ms / num_queries;
    const double avg_query_pq_prefetch_issue_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_pq_prefetch_issue_ms / num_queries;
    const double avg_query_pq_prefetch_wait_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_pq_prefetch_wait_ms / num_queries;
    const double avg_query_pq_checksum_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_pq_checksum_ms / num_queries;
    const double avg_query_queue_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_queue_ms / num_queries;
    const double avg_query_queue_scan_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_queue_scan_ms / num_queries;
    const double avg_query_queue_select_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_queue_select_ms / num_queries;
    const double avg_query_frontier_sort_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_frontier_sort_ms / num_queries;
    const double avg_query_tail_merge_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_tail_merge_ms / num_queries;
    const double avg_query_candidate_sort_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_candidate_sort_ms / num_queries;
    const double avg_query_hash_rebuild_ms =
        num_queries == 0 ? 0.0 : totals.topology_sum_query_hash_rebuild_ms / num_queries;
    const WarmQpsBreakdown warm = BuildWarmQpsBreakdown(totals);
    std::cout << "[topoanns_profile] frontier_pq_mode=" << frontier_pq_mode
              << " top_l=" << top_l
              << " rerank_top_n=" << rerank_top_n
              << " rerank_mode=" << RerankModeName(rerank_mode)
              << " rvq_entry_ms=" << totals.rvq_entry_ms
              << " rvq_kernel_ms=" << totals.rvq_kernel_ms
              << " rvq_entry_gather_ms=" << totals.rvq_entry_gather_ms
              << " rvq_host_expand_ms=" << totals.rvq_host_expand_ms
              << " query_h2d_ms=" << totals.query_h2d_ms
              << " query_h2d_rvq_ms=" << totals.rvq_query_upload_ms
              << " query_h2d_pq_ms=" << totals.pq_query_upload_ms
              << " query_h2d_rerank_ms=" << totals.rerank_query_upload_ms
              << " pq_zero_fill_ms=" << totals.pq_zero_fill_ms
              << " pq_table_kernel_ms=" << totals.pq_table_kernel_ms
              << " pq_table_download_ms=" << totals.pq_table_download_ms
              << " topology_kernel_ms=" << totals.topology_kernel_ms
              << " topology_candidate_download_ms=" << totals.topology_candidate_download_ms
              << " topology_stats_download_ms=" << totals.topology_stats_download_ms
              << " topology_host_postprocess_ms=" << totals.topology_host_postprocess_ms
              << " topology_candidate_materialize_ms="
              << totals.topology_candidate_materialize_ms
              << " topology_topk_extract_ms=" << totals.topology_topk_extract_ms
              << " avg_query_pq_ms=" << avg_query_pq_ms
              << " avg_query_pq_compute_ms=" << avg_query_pq_compute_ms
              << " avg_query_pq_prefetch_issue_ms=" << avg_query_pq_prefetch_issue_ms
              << " avg_query_pq_prefetch_wait_ms=" << avg_query_pq_prefetch_wait_ms
              << " avg_query_pq_checksum_ms=" << avg_query_pq_checksum_ms
              << " avg_query_queue_ms=" << avg_query_queue_ms
              << " avg_query_queue_scan_ms=" << avg_query_queue_scan_ms
              << " avg_query_queue_select_ms=" << avg_query_queue_select_ms
              << " avg_query_frontier_sort_ms=" << avg_query_frontier_sort_ms
              << " avg_query_tail_merge_ms=" << avg_query_tail_merge_ms
              << " avg_query_candidate_sort_ms=" << avg_query_candidate_sort_ms
              << " avg_query_hash_rebuild_ms=" << avg_query_hash_rebuild_ms
              << " rerank_prepare_ms=" << totals.rerank_prepare_ms
              << " rerank_pq2_query_tables_ms=" << totals.rerank_pq2_query_tables_ms
              << " rerank_pq2_refine_kernel_ms=" << totals.rerank_pq2_refine_kernel_ms
              << " rerank_pq2_bound_filter_ms=" << totals.rerank_pq2_bound_filter_ms
              << " rerank_pq2_threshold_update_ms="
              << totals.rerank_pq2_threshold_update_ms
              << " rerank_learned_model_ms=" << totals.rerank_learned_model_ms
              << " rerank_learned_checkpoint_bookkeeping_ms="
              << totals.rerank_learned_checkpoint_bookkeeping_ms
              << " rerank_learned_topk_churn_ms="
              << totals.rerank_learned_topk_churn_ms
              << " rerank_learned_next_window_scan_ms="
              << totals.rerank_learned_next_window_scan_ms
              << " rerank_learned_logit_eval_ms="
              << totals.rerank_learned_logit_eval_ms
              << " io_ms=" << totals.rerank_io_ms
              << " rerank_unpack_ms=" << totals.rerank_unpack_ms
              << " exact_kernel_ms=" << totals.rerank_exact_kernel_ms
              << " rerank_sort_ms=" << totals.rerank_sort_ms
              << " rerank_result_download_ms=" << totals.rerank_result_download_ms
              << " rerank_topk_extract_ms=" << totals.rerank_topk_extract_ms
              << " rerank_sorted_candidates_materialize_ms="
              << totals.rerank_sorted_candidates_materialize_ms
              << " warm_query_h2d_ms=" << warm.query_h2d_ms
              << " warm_rvq_ms=" << warm.rvq_entry_compute_ms
              << " warm_pq_lut_ms=" << warm.pq_lut_compute_ms
              << " warm_topology_ms=" << warm.topology_search_ms
              << " warm_rerank_candidate_ms=" << warm.rerank_candidate_organization_ms
              << " warm_page_read_ms=" << warm.page_read_ms
              << " warm_unpack_ms=" << warm.vector_unpack_ms
              << " warm_exact_ms=" << warm.exact_distance_ms
              << " warm_final_topk_select_ms=" << warm.final_topk_select_ms
              << " warm_final_topk_d2h_ms=" << warm.final_topk_d2h_ms
              << " warm_total_ms=" << warm.total_ms()
              << std::endl;
}

void PrintRerankDebugLine(const std::string& frontier_pq_mode,
                          std::size_t top_l,
                          std::size_t rerank_top_n,
                          topoanns::RerankExecutionMode rerank_mode,
                          std::size_t num_queries,
                          const EvalProfileTotals& totals) {
    const double avg_refine_input =
        num_queries == 0 ? 0.0
                         : static_cast<double>(totals.pq2_refine_input_candidates) /
                               static_cast<double>(num_queries);
    const double avg_requested_top_n =
        totals.rerank_batches == 0
            ? 0.0
            : static_cast<double>(totals.rerank_requested_top_n) /
                  static_cast<double>(totals.rerank_batches);
    const double avg_effective_top_n =
        totals.rerank_batches == 0
            ? 0.0
            : static_cast<double>(totals.rerank_effective_top_n) /
                  static_cast<double>(totals.rerank_batches);
    const double avg_refine_top_l =
        totals.pq2_refine_batches == 0
            ? 0.0
            : static_cast<double>(totals.pq2_refine_top_l) /
                  static_cast<double>(totals.pq2_refine_batches);
    const double avg_tile_size =
        totals.rerank_batches == 0
            ? 0.0
            : static_cast<double>(totals.rerank_tile_size) /
                  static_cast<double>(totals.rerank_batches);
    const std::size_t total_tiles =
        totals.rerank_prepare_nonzero_tiles + totals.rerank_prepare_zero_tiles;
    const double avg_prepare_valid_per_tile =
        total_tiles == 0
            ? 0.0
            : static_cast<double>(totals.rerank_prepare_valid_candidates) /
                  static_cast<double>(total_tiles);
    const double avg_prepare_filtered_per_tile =
        total_tiles == 0
            ? 0.0
            : static_cast<double>(totals.rerank_prepare_filtered_candidates) /
                  static_cast<double>(total_tiles);
    const double avg_learned_stop_prefix =
        num_queries == 0
            ? 0.0
            : static_cast<double>(totals.rerank_learned_stop_prefix_sum) /
                  static_cast<double>(num_queries);
    const double avg_learned_checkpoints =
        num_queries == 0
            ? 0.0
            : static_cast<double>(totals.rerank_learned_checkpoints) /
                  static_cast<double>(num_queries);
    std::cout << "[topoanns_rerank_debug] frontier_pq_mode=" << frontier_pq_mode
              << " top_l=" << top_l
              << " rerank_top_n=" << rerank_top_n
              << " rerank_mode=" << RerankModeName(rerank_mode)
              << " pq2_query_upload_ms=" << totals.rerank_pq2_query_upload_ms
              << " pq2_query_zero_fill_ms=" << totals.rerank_pq2_query_zero_fill_ms
              << " pq2_query_table_kernel_ms=" << totals.rerank_pq2_query_table_kernel_ms
              << " pq2_query_table_download_ms="
              << totals.rerank_pq2_query_table_download_ms
              << " pq2_refine_batches=" << totals.pq2_refine_batches
              << " pq2_residual_batches=" << totals.pq2_residual_refine_batches
              << " avg_valid_candidates_before_refine=" << avg_refine_input
              << " refine_input_min=" << totals.pq2_refine_input_candidates_min
              << " refine_input_max=" << totals.pq2_refine_input_candidates_max
              << " avg_refine_top_l=" << avg_refine_top_l
              << " avg_requested_top_n=" << avg_requested_top_n
              << " avg_effective_top_n=" << avg_effective_top_n
              << " avg_prepare_valid_per_tile=" << avg_prepare_valid_per_tile
              << " prepare_valid_tile_min="
              << totals.rerank_prepare_valid_candidates_min
              << " prepare_valid_tile_max="
              << totals.rerank_prepare_valid_candidates_max
              << " avg_prepare_filtered_per_tile=" << avg_prepare_filtered_per_tile
              << " prepare_filtered_tile_min="
              << totals.rerank_prepare_filtered_candidates_min
              << " prepare_filtered_tile_max="
              << totals.rerank_prepare_filtered_candidates_max
              << " prepare_nonzero_tiles=" << totals.rerank_prepare_nonzero_tiles
              << " prepare_zero_tiles=" << totals.rerank_prepare_zero_tiles
              << " avg_tile_size=" << avg_tile_size
              << " bam_direct_batches=" << totals.rerank_bam_direct_batches
              << " gpu_sort_batches=" << totals.rerank_gpu_sort_batches
              << " host_sort_batches=" << totals.rerank_host_sort_batches
              << " learned_model_ms=" << totals.rerank_learned_model_ms
              << " learned_checkpoint_bookkeeping_ms="
              << totals.rerank_learned_checkpoint_bookkeeping_ms
              << " learned_topk_churn_ms=" << totals.rerank_learned_topk_churn_ms
              << " learned_next_window_scan_ms="
              << totals.rerank_learned_next_window_scan_ms
              << " learned_logit_eval_ms=" << totals.rerank_learned_logit_eval_ms
              << " learned_stop_queries=" << totals.rerank_learned_stop_queries
              << " avg_learned_stop_prefix=" << avg_learned_stop_prefix
              << " learned_stop_prefix_min=" << totals.rerank_learned_stop_prefix_min
              << " learned_stop_prefix_max=" << totals.rerank_learned_stop_prefix_max
              << " avg_learned_checkpoints=" << avg_learned_checkpoints
              << std::endl;
}

void PrintSubbatchTailLine(const std::string& frontier_pq_mode,
                           std::size_t top_l,
                           std::size_t rerank_top_n,
                           topoanns::RerankExecutionMode rerank_mode,
                           std::size_t batch_begin,
                           std::size_t batch_queries,
                           const topoanns::TopologySearchBatchProfile& topology_profile) {
    std::cout << "[topoanns_subbatch_tail]"
              << " frontier_pq_mode=" << frontier_pq_mode
              << " top_l=" << top_l
              << " rerank_top_n=" << rerank_top_n
              << " rerank_mode=" << RerankModeName(rerank_mode)
              << " batch_begin=" << batch_begin
              << " batch_queries=" << batch_queries
              << " max_expanded=" << topology_profile.batch_max_expanded_nodes
              << " p95_expanded=" << topology_profile.batch_p95_expanded_nodes
              << " max_pq_cycles=" << topology_profile.batch_max_pq_cycles
              << " max_queue_cycles=" << topology_profile.batch_max_queue_cycles
              << std::endl;
}

void PrintSubbatchTopologyDetailLine(
    const std::string& frontier_pq_mode,
    std::size_t top_l,
    std::size_t rerank_top_n,
    topoanns::RerankExecutionMode rerank_mode,
    std::size_t batch_begin,
    std::size_t batch_queries,
    const topoanns::TopologySearchBatchProfile& topology_profile) {
    const double avg_candidate_sort_before_full_prefix_ms =
        batch_queries == 0
            ? 0.0
            : topology_profile.sum_query_candidate_sort_before_full_prefix_ms /
                  static_cast<double>(batch_queries);
    const double avg_candidate_sort_after_full_prefix_ms =
        batch_queries == 0
            ? 0.0
            : topology_profile.sum_query_candidate_sort_after_full_prefix_ms /
                  static_cast<double>(batch_queries);
    std::cout << "[topoanns_subbatch_topology_detail]"
              << " frontier_pq_mode=" << frontier_pq_mode
              << " top_l=" << top_l
              << " rerank_top_n=" << rerank_top_n
              << " rerank_mode=" << RerankModeName(rerank_mode)
              << " batch_begin=" << batch_begin
              << " batch_queries=" << batch_queries
              << " avg_iterations=" << topology_profile.batch_avg_iterations
              << " p50_iterations=" << topology_profile.batch_p50_iterations
              << " p95_iterations=" << topology_profile.batch_p95_iterations
              << " max_iterations=" << topology_profile.batch_max_iterations
              << " full_prefix_reached_queries="
              << topology_profile.batch_full_prefix_reached_queries
              << " avg_full_prefix_iteration="
              << topology_profile.batch_avg_full_prefix_iteration
              << " p50_full_prefix_iteration="
              << topology_profile.batch_p50_full_prefix_iteration
              << " p95_full_prefix_iteration="
              << topology_profile.batch_p95_full_prefix_iteration
              << " max_full_prefix_iteration="
              << topology_profile.batch_max_full_prefix_iteration
              << " avg_candidate_sort_before_full_prefix_ms="
              << avg_candidate_sort_before_full_prefix_ms
              << " avg_candidate_sort_after_full_prefix_ms="
              << avg_candidate_sort_after_full_prefix_ms
              << std::endl;
}

void AccumulateCommonProfile(EvalProfileTotals* totals,
                             const topoanns::RvqEntryProfile& entry_profile,
                             const topoanns::PqQueryTablesProfile& pq_profile,
                             const topoanns::TopologySearchBatchProfile& topology_profile) {
    totals->rvq_entry_ms += entry_profile.total_ms;
    totals->rvq_kernel_ms += entry_profile.search_kernel_ms;
    totals->rvq_entry_gather_ms += entry_profile.entry_gather_ms;
    totals->rvq_host_expand_ms +=
        entry_profile.total_ms - entry_profile.query_upload_ms -
        entry_profile.search_kernel_ms - entry_profile.entry_gather_ms;
    totals->rvq_query_upload_ms += entry_profile.query_upload_ms;
    totals->pq_query_upload_ms += pq_profile.query_upload_ms;
    totals->query_h2d_ms += entry_profile.query_upload_ms + pq_profile.query_upload_ms;
    totals->pq_zero_fill_ms += pq_profile.zero_fill_ms;
    totals->pq_table_kernel_ms += pq_profile.kernel_ms;
    totals->pq_table_download_ms += pq_profile.table_download_ms;
    totals->topology_kernel_ms += topology_profile.kernel_ms;
    totals->topology_candidate_download_ms += topology_profile.candidate_download_ms;
    totals->topology_stats_download_ms += topology_profile.stats_download_ms;
    totals->topology_host_postprocess_ms += topology_profile.host_postprocess_ms;
    totals->topology_candidate_materialize_ms += topology_profile.candidate_materialize_ms;
    totals->topology_topk_extract_ms += topology_profile.topology_topk_extract_ms;
    totals->topology_sum_query_pq_ms += topology_profile.sum_query_pq_distance_ms;
    totals->topology_sum_query_pq_compute_ms += topology_profile.sum_query_pq_compute_ms;
    totals->topology_sum_query_pq_prefetch_issue_ms +=
        topology_profile.sum_query_pq_prefetch_issue_ms;
    totals->topology_sum_query_pq_prefetch_wait_ms +=
        topology_profile.sum_query_pq_prefetch_wait_ms;
    totals->topology_sum_query_pq_checksum_ms += topology_profile.sum_query_pq_checksum_ms;
    totals->topology_sum_query_queue_ms += topology_profile.sum_query_queue_update_ms;
    totals->topology_sum_query_queue_scan_ms += topology_profile.sum_query_queue_scan_ms;
    totals->topology_sum_query_queue_select_ms += topology_profile.sum_query_queue_select_ms;
    totals->topology_sum_query_frontier_sort_ms += topology_profile.sum_query_frontier_sort_ms;
    totals->topology_sum_query_tail_merge_ms += topology_profile.sum_query_tail_merge_ms;
    totals->topology_sum_query_candidate_sort_ms +=
        topology_profile.sum_query_candidate_sort_ms;
    totals->topology_sum_query_hash_rebuild_ms += topology_profile.sum_query_hash_rebuild_ms;
}

void AccumulateRerankProfile(EvalProfileTotals* totals,
                             const topoanns::RerankBatchProfile& rerank_profile) {
    totals->rerank_query_upload_ms += rerank_profile.query_upload_ms;
    totals->query_h2d_ms += rerank_profile.query_upload_ms;
    totals->rerank_prepare_ms += rerank_profile.prepare_ms;
    totals->rerank_pq2_query_tables_ms += rerank_profile.pq2_query_tables_ms;
    totals->rerank_pq2_query_upload_ms += rerank_profile.pq2_query_upload_ms;
    totals->rerank_pq2_query_zero_fill_ms += rerank_profile.pq2_query_zero_fill_ms;
    totals->rerank_pq2_query_table_kernel_ms += rerank_profile.pq2_query_table_kernel_ms;
    totals->rerank_pq2_query_table_download_ms +=
        rerank_profile.pq2_query_table_download_ms;
    totals->rerank_pq2_refine_kernel_ms += rerank_profile.pq2_refine_kernel_ms;
    totals->rerank_pq2_bound_filter_ms += rerank_profile.pq2_bound_filter_ms;
    totals->rerank_pq2_threshold_update_ms += rerank_profile.pq2_threshold_update_ms;
    totals->rerank_learned_model_ms += rerank_profile.learned_stop_model_ms;
    totals->rerank_learned_checkpoint_bookkeeping_ms +=
        rerank_profile.learned_stop_checkpoint_bookkeeping_ms;
    totals->rerank_learned_topk_churn_ms +=
        rerank_profile.learned_stop_topk_churn_ms;
    totals->rerank_learned_next_window_scan_ms +=
        rerank_profile.learned_stop_next_window_scan_ms;
    totals->rerank_learned_logit_eval_ms +=
        rerank_profile.learned_stop_logit_eval_ms;
    totals->rerank_io_ms += rerank_profile.io_ms;
    totals->rerank_unpack_ms += rerank_profile.unpack_ms;
    totals->rerank_exact_kernel_ms += rerank_profile.exact_distance_kernel_ms;
    totals->rerank_sort_ms += rerank_profile.sort_ms;
    totals->rerank_result_download_ms += rerank_profile.result_download_ms;
    totals->rerank_topk_extract_ms += rerank_profile.rerank_topk_extract_ms;
    totals->rerank_sorted_candidates_materialize_ms +=
        rerank_profile.sorted_candidates_materialize_ms;
    totals->rerank_batches += rerank_profile.rerank_batches;
    totals->pq2_refine_batches += rerank_profile.pq2_refine_batches;
    totals->pq2_residual_refine_batches += rerank_profile.pq2_residual_refine_batches;
    totals->rerank_bam_direct_batches += rerank_profile.rerank_bam_direct_batches;
    totals->rerank_gpu_sort_batches += rerank_profile.rerank_gpu_sort_batches;
    totals->rerank_host_sort_batches += rerank_profile.rerank_host_sort_batches;
    totals->rerank_requested_top_n += rerank_profile.requested_top_n;
    totals->rerank_effective_top_n += rerank_profile.effective_top_n;
    totals->pq2_refine_top_l += rerank_profile.pq2_refine_top_l;
    totals->pq2_refine_input_candidates += rerank_profile.pq2_refine_input_candidates;
    totals->rerank_tile_size += rerank_profile.rerank_tile_size;
    totals->rerank_prepare_valid_candidates +=
        rerank_profile.rerank_prepare_valid_candidates;
    totals->rerank_prepare_filtered_candidates +=
        rerank_profile.rerank_prepare_filtered_candidates;
    const std::size_t previous_tile_count =
        totals->rerank_prepare_nonzero_tiles + totals->rerank_prepare_zero_tiles;
    totals->rerank_prepare_nonzero_tiles += rerank_profile.rerank_prepare_nonzero_tiles;
    totals->rerank_prepare_zero_tiles += rerank_profile.rerank_prepare_zero_tiles;
    totals->rerank_learned_checkpoints += rerank_profile.learned_stop_checkpoints;
    totals->rerank_learned_stop_queries += rerank_profile.learned_stop_queries;
    totals->rerank_learned_stop_prefix_sum += rerank_profile.learned_stop_prefix_sum;
    bool has_dst_refine = totals->pq2_refine_batches > rerank_profile.pq2_refine_batches;
    MergeMinMaxTotals(rerank_profile.pq2_refine_input_candidates_min,
                      rerank_profile.pq2_refine_input_candidates_max,
                      rerank_profile.pq2_refine_batches != 0,
                      &totals->pq2_refine_input_candidates_min,
                      &totals->pq2_refine_input_candidates_max, &has_dst_refine);
    bool has_dst_valid = previous_tile_count != 0;
    MergeMinMaxTotals(rerank_profile.rerank_prepare_valid_candidates_min,
                      rerank_profile.rerank_prepare_valid_candidates_max,
                      (rerank_profile.rerank_prepare_nonzero_tiles +
                       rerank_profile.rerank_prepare_zero_tiles) != 0,
                      &totals->rerank_prepare_valid_candidates_min,
                      &totals->rerank_prepare_valid_candidates_max, &has_dst_valid);
    bool has_dst_filtered = previous_tile_count != 0;
    MergeMinMaxTotals(rerank_profile.rerank_prepare_filtered_candidates_min,
                      rerank_profile.rerank_prepare_filtered_candidates_max,
                      (rerank_profile.rerank_prepare_nonzero_tiles +
                       rerank_profile.rerank_prepare_zero_tiles) != 0,
                      &totals->rerank_prepare_filtered_candidates_min,
                      &totals->rerank_prepare_filtered_candidates_max, &has_dst_filtered);
    bool has_dst_learned = totals->rerank_batches > rerank_profile.rerank_batches;
    MergeMinMaxTotals(rerank_profile.learned_stop_prefix_min,
                      rerank_profile.learned_stop_prefix_max,
                      rerank_profile.learned_stop_checkpoints != 0,
                      &totals->rerank_learned_stop_prefix_min,
                      &totals->rerank_learned_stop_prefix_max, &has_dst_learned);
}

void AccumulateQueryStats(EvalAggregate* aggregate,
                          const topoanns::TopoAnnsBatchResult& result,
                          const IntMatrix* gt,
                          std::size_t gt_offset,
                          std::size_t top_k,
                          std::size_t batch_queries) {
    aggregate->total_io_pages += result.rerank_stats.io_pages;
    aggregate->total_exact_count += result.rerank_stats.exact_distance_count;
    aggregate->total_bound_filtered += result.rerank_stats.bound_filtered_count;
    for (const auto& query_result : result.queries) {
        aggregate->total_visited += query_result.topology.stats.visited_nodes;
        aggregate->total_expanded += query_result.topology.stats.expanded_nodes;
    }
    if (gt != nullptr) {
        aggregate->recall_sum +=
            RecallAtK(result, *gt, gt_offset, top_k) * static_cast<double>(batch_queries);
    }
}

std::vector<SweepPoint> ComputeParetoFrontier(const std::vector<SweepPoint>& points) {
    std::vector<SweepPoint> ordered = points;
    std::sort(ordered.begin(), ordered.end(), [](const SweepPoint& lhs, const SweepPoint& rhs) {
        if (lhs.recall_at_10 != rhs.recall_at_10) {
            return lhs.recall_at_10 < rhs.recall_at_10;
        }
        return lhs.qps > rhs.qps;
    });
    std::vector<SweepPoint> frontier;
    double best_qps = -1.0;
    for (auto it = ordered.rbegin(); it != ordered.rend(); ++it) {
        if (it->qps > best_qps) {
            frontier.push_back(*it);
            best_qps = it->qps;
        }
    }
    std::reverse(frontier.begin(), frontier.end());
    return frontier;
}

void PrintParetoFrontier(const std::vector<SweepPoint>& points) {
    const std::vector<SweepPoint> frontier = ComputeParetoFrontier(points);
    for (const auto& point : frontier) {
        std::cout << "[topoanns_pareto]"
                  << " frontier_pq_mode=" << point.frontier_pq_mode
                  << " top_l=" << point.top_l
                  << " rerank_top_n=" << point.rerank_top_n
                  << " rerank_mode=" << RerankModeName(point.rerank_mode)
                  << " queries=" << point.num_queries
                  << " qps=" << point.qps
                  << " wall_qps=" << point.wall_qps
                  << " recall@10=" << point.recall_at_10
                  << std::endl;
    }
}

void PrintInitProfile(const InitProfile& profile) {
    const double total_ms = profile.query_load_ms +
                            profile.gt_load_ms +
                            profile.topology_load_ms +
                            profile.pq_load_ms +
                            profile.vector_store_header_ms +
                            profile.bam_provider_init_ms +
                            profile.rvq_load_ms;
    std::cout << "[topoanns_init]"
              << " total_ms=" << total_ms
              << " query_load_ms=" << profile.query_load_ms
              << " gt_load_ms=" << profile.gt_load_ms
              << " topology_load_ms=" << profile.topology_load_ms
              << " pq_load_ms=" << profile.pq_load_ms
              << " vector_store_header_ms=" << profile.vector_store_header_ms
              << " bam_provider_init_ms=" << profile.bam_provider_init_ms
              << " rvq_load_ms=" << profile.rvq_load_ms
              << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        const topoanns::BamRuntimeConfig bam_runtime = topoanns::ResolveBamRuntimeConfig(
            args.bam_config_path, args.bam_overrides, args.allow_bam_controller_override);
        InitProfile init_profile;
        cudaError_t device_status = cudaSetDevice(static_cast<int>(bam_runtime.cuda_device));
        if (device_status != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed for requested cuda device.");
        }
        RequireL40Device(bam_runtime.cuda_device, "topoanns_eval_sift");
        const auto query_load_begin = std::chrono::steady_clock::now();
        const FloatMatrix queries = LoadFloatMatrix(args.query_bin);
        const auto query_load_end = std::chrono::steady_clock::now();
        init_profile.query_load_ms =
            std::chrono::duration<double, std::milli>(query_load_end - query_load_begin).count();
        if (queries.cols == 0 || args.num_queries > queries.rows) {
            throw std::runtime_error("Requested num_queries exceeds query file rows.");
        }
        IntMatrix gt;
        if (args.has_gt) {
            const auto gt_load_begin = std::chrono::steady_clock::now();
            gt = LoadIntMatrix(args.gt_bin);
            const auto gt_load_end = std::chrono::steady_clock::now();
            init_profile.gt_load_ms =
                std::chrono::duration<double, std::milli>(gt_load_end - gt_load_begin).count();
            if (args.num_queries > gt.rows || args.top_k > gt.cols) {
                throw std::runtime_error("GT rows/cols are smaller than requested evaluation.");
            }
        }

        const auto topology_load_begin = std::chrono::steady_clock::now();
        topoanns::SearchResources resources =
            topoanns::SearchResources::FromTopologyFile(args.index_dir / "topology.bin");
        const auto topology_load_end = std::chrono::steady_clock::now();
        init_profile.topology_load_ms =
            std::chrono::duration<double, std::milli>(topology_load_end - topology_load_begin)
                .count();

        const auto pq_load_begin = std::chrono::steady_clock::now();
        resources.LoadPqIndex(args.index_dir / "_pq_pivots.bin",
                              args.index_dir / "_pq_compressed.bin");
        const auto pq_load_end = std::chrono::steady_clock::now();
        init_profile.pq_load_ms =
            std::chrono::duration<double, std::milli>(pq_load_end - pq_load_begin).count();
        if (!args.pq2_pivots.empty()) {
            resources.LoadPq2Index(args.pq2_pivots, args.pq2_codes, args.pq2_error_bounds);
        }

        const auto vector_store_begin = std::chrono::steady_clock::now();
        resources.LoadVectorStore(args.index_dir / "vectors.ssd");
        const auto vector_store_end = std::chrono::steady_clock::now();
        init_profile.vector_store_header_ms =
            std::chrono::duration<double, std::milli>(vector_store_end - vector_store_begin)
                .count();
        topoanns::BamVectorProviderOptions bam_options;
        topoanns::ApplyBamRuntimeConfig(bam_runtime, &bam_options);
        bam_options.device_offset_bytes = args.bam_device_offset_bytes;
        const auto bam_provider_begin = std::chrono::steady_clock::now();
        resources.AttachVectorPageProvider(std::make_shared<topoanns::BamVectorPageProvider>(
            args.index_dir / "vectors.ssd",
            resources.vector_store_layout().header_bytes(),
            resources.vector_store_layout().page_size_bytes(),
            bam_options));
        const auto bam_provider_end = std::chrono::steady_clock::now();
        init_profile.bam_provider_init_ms =
            std::chrono::duration<double, std::milli>(bam_provider_end - bam_provider_begin)
                .count();
        if (resources.vector_store_header().dim != queries.cols) {
            throw std::runtime_error("Query dim and vector store dim do not match.");
        }

        const auto rvq_load_begin = std::chrono::steady_clock::now();
        topoanns::RvqModel rvq_model = topoanns::RvqModel::Load(args.rvq_model);
        const auto rvq_load_end = std::chrono::steady_clock::now();
        init_profile.rvq_load_ms =
            std::chrono::duration<double, std::milli>(rvq_load_end - rvq_load_begin).count();
        if (rvq_model.dim() != queries.cols) {
            throw std::runtime_error("RVQ model dim and query dim do not match.");
        }
        PrintInitProfile(init_profile);

        std::vector<SweepPoint> sweep_points;
        for (const std::string& frontier_pq_mode : args.frontier_pq_modes) {
            ConfigureFrontierPqMode(frontier_pq_mode);
            std::cout << "[topoanns_frontier_pq_mode] mode=" << frontier_pq_mode << std::endl;
            for (std::size_t top_l_index = 0; top_l_index < args.top_l_values.size();
                 ++top_l_index) {
                const std::size_t top_l = args.top_l_values[top_l_index];
                const std::size_t rerank_top_n =
                    args.rerank_top_n_values.empty() ? args.rerank_top_n
                                                     : args.rerank_top_n_values[top_l_index];
                for (const std::size_t eval_queries : args.eval_query_counts) {
                    const std::size_t effective_batch_size =
                        std::min(args.batch_size, eval_queries);
                    const bool sweep_rerank =
                        !args.rerank_top_n_sweep_values.empty() ||
                        !args.rerank_mode_sweep_values.empty();
                    if (!sweep_rerank) {
                    topoanns::TopoAnnsSearchParams params;
                    params.topology.top_k = std::max(args.top_k, rerank_top_n);
                    params.topology.top_l = top_l;
                    params.topology.search_width = args.search_width;
                    params.topology.max_expansions = args.max_expansions;
                    params.rerank.top_k = args.top_k;
                    params.rerank.top_n = rerank_top_n;
                    params.rerank.mode = args.rerank_mode;
                    params.rerank.rank_tile_size = args.rerank_rank_tile_size;
                    params.rerank.use_pq2_refine = args.use_pq2_refine;
                    params.rerank.use_pq2_bound_filter = args.use_pq2_bound_filter;
                    params.rerank.pq2_refine_top_l = top_l;
                    params.rerank.use_early_stop = args.use_early_stop;
                    params.rerank.use_learned_stop = args.use_learned_stop;
                    params.rerank.learned_stop_model_path = args.rerank_learned_stop_model;
                    params.rerank.early_stop_min_prefix =
                        args.rerank_early_stop_min_prefix;
                    params.rerank.early_stop_patience_tiles =
                        args.rerank_early_stop_patience_tiles;

                    EvalAggregate aggregate;
                    const auto start = std::chrono::steady_clock::now();
                    for (std::size_t query_offset = 0; query_offset < eval_queries;
                         query_offset += effective_batch_size) {
                        const std::size_t batch_queries =
                            std::min(effective_batch_size, eval_queries - query_offset);
                        const float* begin =
                            queries.values.data() +
                            query_offset * static_cast<std::size_t>(queries.cols);
                        const float* end =
                            begin + batch_queries * static_cast<std::size_t>(queries.cols);
                        std::vector<float> batch_query_buffer(begin, end);

                        topoanns::RvqEntryProfile entry_profile;
                        topoanns::DeviceEntryBatch entry_batch =
                            rvq_model.ComputeFloat32DeviceEntryBatch(
                                batch_query_buffer, batch_queries, args.rvq_entry_count, 0,
                                &entry_profile);
                        const topoanns::TopoAnnsBatchResult result =
                            topoanns::TopoAnnsSearch::RunBatchFloat32Fused(resources,
                                                                           entry_batch,
                                                                           batch_query_buffer,
                                                                           batch_queries,
                                                                           params);
                        PrintSubbatchTailLine(frontier_pq_mode, top_l, rerank_top_n, args.rerank_mode, query_offset,
                                              batch_queries, result.topology_profile);
                        PrintSubbatchTopologyDetailLine(frontier_pq_mode, top_l, rerank_top_n,
                                                        args.rerank_mode, query_offset,
                                                        batch_queries,
                                                        result.topology_profile);
                        AccumulateCommonProfile(&aggregate.profile_totals, entry_profile,
                                                result.pq_profile, result.topology_profile);
                        AccumulateRerankProfile(&aggregate.profile_totals, result.rerank_profile);
                        AccumulateQueryStats(&aggregate, result, args.has_gt ? &gt : nullptr,
                                             query_offset, args.top_k, batch_queries);

                        std::cout << "[topoanns_eval] frontier_pq_mode=" << frontier_pq_mode
                                  << " top_l=" << top_l
                                  << " rerank_top_n=" << rerank_top_n
                                  << " finished queries " << (query_offset + batch_queries)
                                  << " / " << eval_queries << std::endl;
                    }

                    const auto end = std::chrono::steady_clock::now();
                    const double elapsed_sec =
                        std::chrono::duration<double>(end - start).count();
                    const double recall_at_10 =
                        args.has_gt
                            ? aggregate.recall_sum / static_cast<double>(eval_queries)
                            : -1.0;
                    const WarmQpsBreakdown warm_breakdown =
                        BuildWarmQpsBreakdown(aggregate.profile_totals);
                    PrintSummaryLine(frontier_pq_mode,
                                     top_l,
                                     rerank_top_n,
                                     args.rerank_mode,
                                     eval_queries,
                                     elapsed_sec,
                                     warm_breakdown,
                                     recall_at_10,
                                     aggregate.total_io_pages,
                                     aggregate.total_exact_count,
                                     aggregate.total_bound_filtered,
                                     aggregate.total_visited,
                                     aggregate.total_expanded);
                    PrintProfileLine(frontier_pq_mode, top_l, rerank_top_n, args.rerank_mode, eval_queries,
                                     aggregate.profile_totals);
                    PrintRerankDebugLine(frontier_pq_mode, top_l, rerank_top_n, args.rerank_mode, eval_queries,
                                         aggregate.profile_totals);
                    const double warm_qps =
                        QpsFromMs(eval_queries, warm_breakdown.total_ms());
                    sweep_points.push_back(SweepPoint{
                        frontier_pq_mode,
                        top_l,
                        rerank_top_n,
                        args.rerank_mode,
                        eval_queries,
                        warm_qps,
                        warm_qps,
                        recall_at_10,
                    });
                } else {
                    topoanns::TopologySearchParams topology_params;
                    const std::vector<std::size_t> sweep_top_n_values =
                        args.rerank_top_n_sweep_values.empty()
                            ? std::vector<std::size_t>{rerank_top_n}
                            : args.rerank_top_n_sweep_values;
                    topology_params.top_k =
                        std::max(args.top_k,
                                 *std::max_element(sweep_top_n_values.begin(),
                                                   sweep_top_n_values.end()));
                    topology_params.top_l = top_l;
                    topology_params.search_width = args.search_width;
                    topology_params.max_expansions = args.max_expansions;

                    std::vector<topoanns::RerankExactParams> rerank_params_list;
                    const std::vector<topoanns::RerankExecutionMode> sweep_modes =
                        args.rerank_mode_sweep_values.empty()
                            ? std::vector<topoanns::RerankExecutionMode>{args.rerank_mode}
                            : args.rerank_mode_sweep_values;
                    rerank_params_list.reserve(sweep_top_n_values.size() * sweep_modes.size());
                    for (const std::size_t sweep_top_n : sweep_top_n_values) {
                        if (sweep_top_n > top_l) {
                            continue;
                        }
                        for (const auto sweep_mode : sweep_modes) {
                            topoanns::RerankExactParams rerank_params;
                            rerank_params.top_k = args.top_k;
                            rerank_params.top_n = sweep_top_n;
                            rerank_params.mode = sweep_mode;
                            rerank_params.rank_tile_size = args.rerank_rank_tile_size;
                            rerank_params.use_pq2_refine = args.use_pq2_refine;
                            rerank_params.use_pq2_bound_filter = args.use_pq2_bound_filter;
                            rerank_params.pq2_refine_top_l = top_l;
                            rerank_params.use_early_stop = args.use_early_stop;
                            rerank_params.use_learned_stop = args.use_learned_stop;
                            rerank_params.learned_stop_model_path =
                                args.rerank_learned_stop_model;
                            rerank_params.early_stop_min_prefix =
                                args.rerank_early_stop_min_prefix;
                            rerank_params.early_stop_patience_tiles =
                                args.rerank_early_stop_patience_tiles;
                            rerank_params_list.push_back(rerank_params);
                        }
                    }
                    if (rerank_params_list.empty()) {
                        throw std::runtime_error(
                            "No valid rerank sweep combinations remain after filtering top_n > top_l.");
                    }

                    std::vector<EvalAggregate> aggregates(rerank_params_list.size());
                    const auto start = std::chrono::steady_clock::now();
                    for (std::size_t query_offset = 0; query_offset < eval_queries;
                         query_offset += effective_batch_size) {
                        const std::size_t batch_queries =
                            std::min(effective_batch_size, eval_queries - query_offset);
                        const float* begin =
                            queries.values.data() +
                            query_offset * static_cast<std::size_t>(queries.cols);
                        const float* end =
                            begin + batch_queries * static_cast<std::size_t>(queries.cols);
                        std::vector<float> batch_query_buffer(begin, end);

                        topoanns::RvqEntryProfile entry_profile;
                        topoanns::DeviceEntryBatch entry_batch =
                            rvq_model.ComputeFloat32DeviceEntryBatch(
                                batch_query_buffer, batch_queries, args.rvq_entry_count, 0,
                                &entry_profile);
                        const std::vector<topoanns::TopoAnnsBatchResult> sweep_results =
                            topoanns::TopoAnnsSearch::RunBatchFloat32FusedRerankSweep(
                                resources, entry_batch, batch_query_buffer, batch_queries,
                                topology_params, rerank_params_list);

                        for (std::size_t sweep_index = 0; sweep_index < sweep_results.size();
                             ++sweep_index) {
                            const auto& result = sweep_results[sweep_index];
                            auto& aggregate = aggregates[sweep_index];
                            PrintSubbatchTailLine(frontier_pq_mode, top_l, rerank_params_list[sweep_index].top_n,
                                                  rerank_params_list[sweep_index].mode,
                                                  query_offset, batch_queries,
                                                  result.topology_profile);
                            PrintSubbatchTopologyDetailLine(
                                frontier_pq_mode, top_l, rerank_params_list[sweep_index].top_n,
                                rerank_params_list[sweep_index].mode, query_offset,
                                batch_queries, result.topology_profile);
                            AccumulateCommonProfile(&aggregate.profile_totals, entry_profile,
                                                    result.pq_profile, result.topology_profile);
                            AccumulateRerankProfile(&aggregate.profile_totals,
                                                    result.rerank_profile);
                            AccumulateQueryStats(&aggregate, result, args.has_gt ? &gt : nullptr,
                                                 query_offset, args.top_k, batch_queries);
                        }

                        std::cout << "[topoanns_eval_sweep] frontier_pq_mode=" << frontier_pq_mode
                                  << " top_l=" << top_l
                                  << " rerank_values=" << rerank_params_list.size()
                                  << " finished queries " << (query_offset + batch_queries)
                                  << " / " << eval_queries << std::endl;
                    }

                    for (std::size_t sweep_index = 0; sweep_index < rerank_params_list.size();
                         ++sweep_index) {
                        const std::size_t sweep_top_n = rerank_params_list[sweep_index].top_n;
                        const auto sweep_mode = rerank_params_list[sweep_index].mode;
                        const auto& aggregate = aggregates[sweep_index];
                        const double recall_at_10 =
                            args.has_gt
                                ? aggregate.recall_sum / static_cast<double>(eval_queries)
                                : -1.0;
                        const WarmQpsBreakdown warm_breakdown =
                            BuildWarmQpsBreakdown(aggregate.profile_totals);
                        const double estimated_elapsed_sec =
                            warm_breakdown.total_ms() / 1000.0;
                        PrintSummaryLine(frontier_pq_mode,
                                         top_l,
                                         sweep_top_n,
                                         sweep_mode,
                                         eval_queries,
                                         estimated_elapsed_sec,
                                         warm_breakdown,
                                         recall_at_10,
                                         aggregate.total_io_pages,
                                         aggregate.total_exact_count,
                                         aggregate.total_bound_filtered,
                                         aggregate.total_visited,
                                         aggregate.total_expanded);
                        PrintProfileLine(frontier_pq_mode,
                                         top_l,
                                         sweep_top_n,
                                         sweep_mode,
                                         eval_queries,
                                         aggregate.profile_totals);
                        PrintRerankDebugLine(frontier_pq_mode,
                                             top_l,
                                             sweep_top_n,
                                             sweep_mode,
                                             eval_queries,
                                             aggregate.profile_totals);
                        const double warm_qps =
                            QpsFromMs(eval_queries, warm_breakdown.total_ms());
                        sweep_points.push_back(SweepPoint{
                            frontier_pq_mode,
                            top_l,
                            sweep_top_n,
                            sweep_mode,
                            eval_queries,
                            warm_qps,
                            warm_qps,
                            recall_at_10,
                        });
                    }
                    }
                }
            }
        }
        if (!sweep_points.empty()) {
            PrintParetoFrontier(sweep_points);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_eval] " << e.what() << std::endl;
        return 1;
    }
}
