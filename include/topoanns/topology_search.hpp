#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "topoanns/device_entry_batch.hpp"
#include "topoanns/entry_provider.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/search_stop_condition.hpp"

namespace topoanns {

class PqDistanceOracle;

struct TopologySearchParams {
    std::size_t top_k = 10;
    std::size_t top_l = 32;
    std::size_t candidate_queue_size = 0;
    std::size_t search_width = 1;
    std::size_t max_expansions = 1024;
    std::uint64_t topology_cached_node_count = std::numeric_limits<std::uint64_t>::max();
    bool enable_exact_reuse = false;
    const float* exact_reuse_device_queries = nullptr;
    std::size_t exact_reuse_query_dim = 0;
    std::uint32_t combined_node_bytes = 0;
    std::uint32_t combined_nodes_per_page = 0;
    std::size_t exact_reuse_cache_capacity = 0;
    bool enable_occupancy_profile = false;
    bool enable_expanded_trace = false;
};

struct TopologySearchStats {
    std::size_t visited_nodes = 0;
    std::size_t expanded_nodes = 0;
    std::size_t topology_io_pages = 0;
    std::size_t iterations = 0;
};

struct TopologySearchBatchProfile {
    double kernel_ms = 0.0;
    double candidate_download_ms = 0.0;
    double stats_download_ms = 0.0;
    double host_postprocess_ms = 0.0;
    double candidate_materialize_ms = 0.0;
    double topology_topk_extract_ms = 0.0;
    double sum_query_pq_distance_ms = 0.0;
    double sum_query_pq_compute_ms = 0.0;
    double sum_query_pq_prefetch_issue_ms = 0.0;
    double sum_query_pq_prefetch_wait_ms = 0.0;
    double sum_query_pq_checksum_ms = 0.0;
    double sum_query_queue_update_ms = 0.0;
    double sum_query_queue_scan_ms = 0.0;
    double sum_query_queue_select_ms = 0.0;
    double sum_query_frontier_sort_ms = 0.0;
    double sum_query_tail_merge_ms = 0.0;
    double sum_query_candidate_sort_ms = 0.0;
    double sum_query_candidate_sort_before_full_prefix_ms = 0.0;
    double sum_query_candidate_sort_after_full_prefix_ms = 0.0;
    double sum_query_hash_rebuild_ms = 0.0;
    double sum_query_learned_stop_model_ms = 0.0;
    double sum_query_learned_stop_feature_ms = 0.0;
    double sum_query_learned_stop_find_first_set_ms = 0.0;
    double sum_query_learned_stop_count_bits_ms = 0.0;
    double sum_query_learned_stop_topk_churn_ms = 0.0;
    double sum_query_learned_stop_logit_eval_ms = 0.0;
    double sum_query_combined_node_read_ms = 0.0;
    double sum_query_exact_reuse_insert_ms = 0.0;
    std::uint64_t batch_max_pq_cycles = 0;
    std::uint64_t batch_max_queue_cycles = 0;
    double batch_avg_iterations = 0.0;
    std::size_t batch_p50_iterations = 0;
    std::size_t batch_p95_iterations = 0;
    std::size_t batch_max_iterations = 0;
    std::size_t batch_full_prefix_reached_queries = 0;
    double batch_avg_full_prefix_iteration = 0.0;
    std::size_t batch_p50_full_prefix_iteration = 0;
    std::size_t batch_p95_full_prefix_iteration = 0;
    std::size_t batch_max_full_prefix_iteration = 0;
    std::size_t batch_max_expanded_nodes = 0;
    std::size_t batch_p95_expanded_nodes = 0;
};

struct TopologySearchResult {
    std::vector<RankedCandidate> sorted_candidates;
    std::vector<RankedCandidate> topk;
    TopologySearchStats stats;
};

class TopologySearch {
public:
    static TopologySearchResult Run(const SearchResources& resources,
                                    const EntryProvider& entry_provider,
                                    const PqDistanceOracle& distance_oracle,
                                    std::size_t query_id,
                                    const TopologySearchParams& params,
                                    TopologySearchBatchProfile* out_profile = nullptr);
    static std::vector<TopologySearchResult> RunBatch(const SearchResources& resources,
                                                      const EntryProvider& entry_provider,
                                                      const PqDistanceOracle& distance_oracle,
                                                      const TopologySearchParams& params,
                                                      TopologySearchBatchProfile* out_profile =
                                                          nullptr);
    static std::vector<TopologySearchResult> RunBatch(const SearchResources& resources,
                                                      const DeviceEntryBatch& entry_batch,
                                                      const PqDistanceOracle& distance_oracle,
                                                      const TopologySearchParams& params,
                                                      TopologySearchBatchProfile* out_profile =
                                                          nullptr);
};

}  // namespace topoanns
