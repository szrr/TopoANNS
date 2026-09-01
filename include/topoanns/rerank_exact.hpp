#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "topoanns/common.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/topology_search.hpp"

namespace topoanns {

enum class RerankExecutionMode {
    kLinear,
    kTiled,
    kPageByPage,
    kPersistent,
};

struct RerankExactParams {
    std::size_t top_k = 10;
    std::size_t top_n = 100;
    RerankExecutionMode mode = RerankExecutionMode::kPersistent;
    std::size_t rank_tile_size = 16;
    bool use_pq2_refine = false;
    bool use_pq2_bound_filter = false;
    std::size_t pq2_refine_top_l = 0;
    bool use_early_stop = false;
    std::size_t early_stop_min_prefix = 32;
    std::size_t early_stop_patience_tiles = 1;
    bool use_learned_stop = false;
    std::filesystem::path learned_stop_model_path;
    // Zero keeps the original vector-store page layout.
    std::size_t ssd_records_per_page = 0;
    std::size_t ssd_record_stride_bytes = 0;
};

struct RerankQueryResult {
    std::vector<RankedCandidate> sorted_candidates;
    std::vector<RankedCandidate> topk;
};

struct RerankBatchStats {
    // Counts page requests submitted by the rerank path to the provider.
    std::size_t io_pages = 0;
    std::size_t exact_distance_count = 0;
    std::size_t bound_filtered_count = 0;
};

struct RerankBatchProfile {
    double total_ms = 0.0;
    double pq2_query_tables_ms = 0.0;
    double pq2_query_upload_ms = 0.0;
    double pq2_query_zero_fill_ms = 0.0;
    double pq2_query_table_kernel_ms = 0.0;
    double pq2_query_table_download_ms = 0.0;
    double pq2_refine_kernel_ms = 0.0;
    double pq2_bound_filter_ms = 0.0;
    double pq2_threshold_update_ms = 0.0;
    double learned_stop_model_ms = 0.0;
    double learned_stop_checkpoint_bookkeeping_ms = 0.0;
    double learned_stop_topk_churn_ms = 0.0;
    double learned_stop_next_window_scan_ms = 0.0;
    double learned_stop_logit_eval_ms = 0.0;
    // Sum of complete per-query persistent-rerank block cycles, converted to ms.
    // This has the same aggregation basis as learned_stop_model_ms.
    double rerank_query_block_ms = 0.0;
    double query_upload_ms = 0.0;
    double prepare_ms = 0.0;
    double io_ms = 0.0;
    double unpack_ms = 0.0;
    double exact_distance_total_ms = 0.0;
    double exact_distance_kernel_ms = 0.0;
    double sort_ms = 0.0;
    double result_download_ms = 0.0;
    double rerank_topk_extract_ms = 0.0;
    double sorted_candidates_materialize_ms = 0.0;
    std::size_t pq2_refine_input_candidates = 0;
    std::size_t pq2_refine_input_candidates_min = 0;
    std::size_t pq2_refine_input_candidates_max = 0;
    std::size_t pq2_refine_top_l = 0;
    std::size_t requested_top_n = 0;
    std::size_t effective_top_n = 0;
    std::size_t exact_candidates = 0;
    std::size_t bound_filtered_candidates = 0;
    std::size_t rerank_batches = 0;
    std::size_t pq2_refine_batches = 0;
    std::size_t rerank_tile_size = 0;
    std::size_t rerank_prepare_valid_candidates = 0;
    std::size_t rerank_prepare_valid_candidates_min = 0;
    std::size_t rerank_prepare_valid_candidates_max = 0;
    std::size_t rerank_prepare_filtered_candidates = 0;
    std::size_t rerank_prepare_filtered_candidates_min = 0;
    std::size_t rerank_prepare_filtered_candidates_max = 0;
    std::size_t rerank_prepare_nonzero_tiles = 0;
    std::size_t rerank_prepare_zero_tiles = 0;
    std::size_t pq2_residual_refine_batches = 0;
    std::size_t rerank_bam_direct_batches = 0;
    std::size_t rerank_gpu_sort_batches = 0;
    std::size_t rerank_host_sort_batches = 0;
    std::size_t learned_stop_checkpoints = 0;
    std::size_t learned_stop_queries = 0;
    std::size_t learned_stop_prefix_sum = 0;
    std::size_t learned_stop_prefix_min = 0;
    std::size_t learned_stop_prefix_max = 0;
};

struct RerankBatchResult {
    std::vector<RerankQueryResult> queries;
    RerankBatchStats stats;
    RerankBatchProfile profile;
};

class RerankExact {
public:
    static RerankBatchResult RunBatchFloat32(
        const SearchResources& resources,
        const std::vector<TopologySearchResult>& topology_results,
        const std::vector<float>& queries,
        std::size_t num_queries,
        const RerankExactParams& params,
        RerankBatchProfile* out_profile = nullptr);

    static RerankBatchResult RunBatchUint8(
        const SearchResources& resources,
        const std::vector<TopologySearchResult>& topology_results,
        const std::vector<std::uint8_t>& queries,
        std::size_t num_queries,
        const RerankExactParams& params,
        RerankBatchProfile* out_profile = nullptr);

    static RerankBatchResult RunBatchInt8(
        const SearchResources& resources,
        const std::vector<TopologySearchResult>& topology_results,
        const std::vector<std::int8_t>& queries,
        std::size_t num_queries,
        const RerankExactParams& params,
        RerankBatchProfile* out_profile = nullptr);
};

}  // namespace topoanns
