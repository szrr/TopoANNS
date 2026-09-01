#include "topoanns/topoanns_search.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "fused_rerank_device.hpp"
#include "topoanns/pq_distance_oracle.hpp"
#include "topology_search_kernel.hpp"

namespace topoanns {
namespace {

constexpr std::uint32_t kInvalidFullPrefixIteration = 0xffffffffU;

struct BatchDistributionSummary {
    double avg = 0.0;
    std::size_t p50 = 0;
    std::size_t p95 = 0;
    std::size_t max = 0;
};

BatchDistributionSummary SummarizeDistribution(std::vector<std::size_t> values) {
    BatchDistributionSummary summary;
    if (values.empty()) {
        return summary;
    }
    std::sort(values.begin(), values.end());
    std::size_t total = 0;
    for (std::size_t value : values) {
        total += value;
    }
    const auto rank_index = [&values](std::size_t numerator, std::size_t denominator) {
        const std::size_t rank =
            (numerator * values.size() + denominator - 1) / denominator;
        return std::min(values.size() - 1, std::max<std::size_t>(1, rank) - 1);
    };
    summary.avg = static_cast<double>(total) / static_cast<double>(values.size());
    summary.p50 = values[rank_index(50, 100)];
    summary.p95 = values[rank_index(95, 100)];
    summary.max = values.back();
    return summary;
}

struct FusedTopologyArtifacts {
    PqQueryTablesProfile pq_profile;
    TopologySearchBatchProfile topology_profile;
    std::vector<detail::DeviceTopologySearchStats> host_stats;
    detail::DeviceTopologyBatchResult topology_device;
};

FusedTopologyArtifacts BuildFusedTopologyArtifacts(
    const SearchResources& resources,
    const DeviceEntryBatch& entry_batch,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const TopologySearchParams& topology_params,
    std::size_t required_candidate_queue_size) {
    if (entry_batch.num_queries != num_queries) {
        throw std::runtime_error(BuildErrorMessage(
            "BuildFusedTopologyArtifacts", "entry batch size must equal num_queries."));
    }
    if (!resources.has_pq_index()) {
        throw std::runtime_error(BuildErrorMessage(
            "BuildFusedTopologyArtifacts", "PQ index must be loaded before search."));
    }
    if (!resources.has_vector_store()) {
        throw std::runtime_error(BuildErrorMessage(
            "BuildFusedTopologyArtifacts", "Vector store must be loaded before rerank."));
    }

    const PqDistanceOracle oracle =
        PqDistanceOracle::FromFloatQueries(resources, queries, num_queries);

    TopologySearchParams effective_topology = topology_params;
    effective_topology.candidate_queue_size =
        std::max({effective_topology.top_l,
                  effective_topology.candidate_queue_size,
                  required_candidate_queue_size});

    FusedTopologyArtifacts artifacts;
    artifacts.pq_profile = oracle.query_tables().profile();
    artifacts.topology_device = detail::RunTopologySearchKernelBatchDevice(
        resources, oracle, entry_batch, effective_topology);

    artifacts.topology_profile.kernel_ms = artifacts.topology_device.kernel_ms;
    const auto stats_download_begin = std::chrono::steady_clock::now();
    artifacts.host_stats = artifacts.topology_device.stats_buffer.CopyToHost();
    const std::vector<detail::DeviceTopologyProfileCycles> host_profile =
        artifacts.topology_device.profile_buffer.CopyToHost();
    const auto stats_download_end = std::chrono::steady_clock::now();
    artifacts.topology_profile.stats_download_ms =
        std::chrono::duration<double, std::milli>(stats_download_end - stats_download_begin)
            .count();

    int device = 0;
    ThrowIfCudaError(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp props;
    ThrowIfCudaError(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties");
    const double cycles_per_ms = static_cast<double>(props.clockRate);
    std::vector<std::size_t> expanded_nodes;
    std::vector<std::size_t> iterations;
    std::vector<std::size_t> full_prefix_iterations;
    expanded_nodes.reserve(artifacts.host_stats.size());
    iterations.reserve(artifacts.host_stats.size());
    full_prefix_iterations.reserve(artifacts.host_stats.size());
    for (const auto& query_stats : artifacts.host_stats) {
        expanded_nodes.push_back(query_stats.expanded_nodes);
        iterations.push_back(query_stats.iterations);
        artifacts.topology_profile.batch_max_expanded_nodes =
            std::max<std::size_t>(artifacts.topology_profile.batch_max_expanded_nodes,
                                  query_stats.expanded_nodes);
        if (query_stats.first_full_prefix_iteration != kInvalidFullPrefixIteration) {
            full_prefix_iterations.push_back(query_stats.first_full_prefix_iteration);
        }
    }
    if (!expanded_nodes.empty()) {
        std::sort(expanded_nodes.begin(), expanded_nodes.end());
        const std::size_t p95_rank =
            (95 * expanded_nodes.size() + 99) / 100;
        const std::size_t p95_index =
            std::min(expanded_nodes.size() - 1, std::max<std::size_t>(1, p95_rank) - 1);
        artifacts.topology_profile.batch_p95_expanded_nodes = expanded_nodes[p95_index];
    }
    const BatchDistributionSummary iteration_summary = SummarizeDistribution(iterations);
    artifacts.topology_profile.batch_avg_iterations = iteration_summary.avg;
    artifacts.topology_profile.batch_p50_iterations = iteration_summary.p50;
    artifacts.topology_profile.batch_p95_iterations = iteration_summary.p95;
    artifacts.topology_profile.batch_max_iterations = iteration_summary.max;
    artifacts.topology_profile.batch_full_prefix_reached_queries = full_prefix_iterations.size();
    const BatchDistributionSummary full_prefix_summary =
        SummarizeDistribution(full_prefix_iterations);
    artifacts.topology_profile.batch_avg_full_prefix_iteration = full_prefix_summary.avg;
    artifacts.topology_profile.batch_p50_full_prefix_iteration = full_prefix_summary.p50;
    artifacts.topology_profile.batch_p95_full_prefix_iteration = full_prefix_summary.p95;
    artifacts.topology_profile.batch_max_full_prefix_iteration = full_prefix_summary.max;
    for (const auto& query_profile : host_profile) {
        artifacts.topology_profile.batch_max_pq_cycles =
            std::max<std::uint64_t>(artifacts.topology_profile.batch_max_pq_cycles,
                                    query_profile.pq_cycles);
        artifacts.topology_profile.batch_max_queue_cycles =
            std::max<std::uint64_t>(artifacts.topology_profile.batch_max_queue_cycles,
                                    query_profile.queue_cycles);
        artifacts.topology_profile.sum_query_pq_distance_ms +=
            static_cast<double>(query_profile.pq_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_pq_compute_ms +=
            static_cast<double>(query_profile.pq_compute_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_pq_prefetch_issue_ms +=
            static_cast<double>(query_profile.pq_prefetch_issue_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_pq_prefetch_wait_ms +=
            static_cast<double>(query_profile.pq_prefetch_wait_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_pq_checksum_ms +=
            static_cast<double>(query_profile.pq_checksum_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_queue_update_ms +=
            static_cast<double>(query_profile.queue_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_queue_scan_ms +=
            static_cast<double>(query_profile.queue_scan_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_queue_select_ms +=
            static_cast<double>(query_profile.queue_select_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_frontier_sort_ms +=
            static_cast<double>(query_profile.frontier_sort_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_tail_merge_ms +=
            static_cast<double>(query_profile.tail_merge_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_candidate_sort_ms +=
            static_cast<double>(query_profile.candidate_sort_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_candidate_sort_before_full_prefix_ms +=
            static_cast<double>(query_profile.candidate_sort_before_full_prefix_cycles) /
            cycles_per_ms;
        artifacts.topology_profile.sum_query_candidate_sort_after_full_prefix_ms +=
            static_cast<double>(query_profile.candidate_sort_after_full_prefix_cycles) /
            cycles_per_ms;
        artifacts.topology_profile.sum_query_hash_rebuild_ms +=
            static_cast<double>(query_profile.hash_rebuild_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_learned_stop_model_ms +=
            static_cast<double>(query_profile.learned_stop_model_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_learned_stop_feature_ms +=
            static_cast<double>(query_profile.learned_stop_feature_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_learned_stop_find_first_set_ms +=
            static_cast<double>(query_profile.learned_stop_find_first_set_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_learned_stop_count_bits_ms +=
            static_cast<double>(query_profile.learned_stop_count_bits_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_learned_stop_topk_churn_ms +=
            static_cast<double>(query_profile.learned_stop_topk_churn_cycles) / cycles_per_ms;
        artifacts.topology_profile.sum_query_learned_stop_logit_eval_ms +=
            static_cast<double>(query_profile.learned_stop_logit_eval_cycles) / cycles_per_ms;
    }

    return artifacts;
}

TopoAnnsBatchResult BuildFusedBatchResult(
    std::size_t num_queries,
    const PqQueryTablesProfile& pq_profile,
    const TopologySearchBatchProfile& topology_profile,
    const std::vector<detail::DeviceTopologySearchStats>& host_stats,
    RerankBatchResult rerank_results,
    const RerankBatchProfile& rerank_profile) {
    TopoAnnsBatchResult result;
    result.queries.resize(num_queries);
    result.rerank_stats = rerank_results.stats;
    result.pq_profile = pq_profile;
    result.topology_profile = topology_profile;
    result.rerank_profile = rerank_profile;
    for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
        result.queries[query_id].topology.stats.visited_nodes = host_stats[query_id].visited_nodes;
        result.queries[query_id].topology.stats.expanded_nodes =
            host_stats[query_id].expanded_nodes;
        result.queries[query_id].topology.stats.iterations = host_stats[query_id].iterations;
        result.queries[query_id].rerank = std::move(rerank_results.queries[query_id]);
    }
    return result;
}

template <typename QueryT, typename OracleFactory, typename TopologyRunner,
          typename RerankRunner>
TopoAnnsBatchResult RunBatchImpl(const SearchResources& resources,
                                 const std::vector<QueryT>& queries,
                                 std::size_t num_queries,
                                 const TopoAnnsSearchParams& params,
                                 OracleFactory&& oracle_factory,
                                 TopologyRunner&& topology_runner,
                                 RerankRunner&& rerank_runner,
                                 const char* context) {
    if (!resources.has_pq_index()) {
        throw std::runtime_error(
            BuildErrorMessage(context, "PQ index must be loaded before search."));
    }
    if (!resources.has_vector_store()) {
        throw std::runtime_error(
            BuildErrorMessage(context, "Vector store must be loaded before rerank."));
    }
    if (resources.pq_index().host().ndims != resources.vector_store_header().dim) {
        throw std::runtime_error(BuildErrorMessage(
            context, "PQ ndims must match vector store dim for end-to-end search."));
    }
    if (queries.size() !=
        num_queries * static_cast<std::size_t>(resources.pq_index().host().ndims)) {
        throw std::runtime_error(BuildErrorMessage(
            context, "Query buffer size must equal num_queries * pq_ndims."));
    }

    const PqDistanceOracle oracle = oracle_factory();
    const PqQueryTablesProfile pq_profile = oracle.query_tables().profile();
    TopoAnnsSearchParams effective_params = params;
    effective_params.topology.candidate_queue_size =
        std::max({effective_params.topology.top_l,
                  effective_params.topology.candidate_queue_size,
                  effective_params.rerank.top_n});
    TopologySearchBatchProfile topology_profile;
    const std::vector<TopologySearchResult> topology_results =
        topology_runner(oracle, effective_params, &topology_profile);
    RerankBatchProfile rerank_profile;
    const RerankBatchResult rerank_results =
        rerank_runner(topology_results, queries, num_queries, effective_params.rerank,
                      &rerank_profile);

    TopoAnnsBatchResult result;
    result.queries.resize(num_queries);
    result.rerank_stats = rerank_results.stats;
    result.pq_profile = pq_profile;
    result.topology_profile = topology_profile;
    result.rerank_profile = rerank_profile;
    for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
        result.queries[query_id].topology = topology_results[query_id];
        result.queries[query_id].rerank = rerank_results.queries[query_id];
    }
    return result;
}

}  // namespace

TopoAnnsBatchResult TopoAnnsSearch::RunBatchFloat32(
    const SearchResources& resources,
    const EntryProvider& entry_provider,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const TopoAnnsSearchParams& params) {
    return RunBatchImpl(
        resources, queries, num_queries, params,
        [&]() { return PqDistanceOracle::FromFloatQueries(resources, queries, num_queries); },
        [&](const PqDistanceOracle& oracle,
            const TopoAnnsSearchParams& effective_params,
            TopologySearchBatchProfile* topology_profile) {
            return TopologySearch::RunBatch(resources, entry_provider, oracle,
                                            effective_params.topology, topology_profile);
        },
        [&](const std::vector<TopologySearchResult>& topology_results,
            const std::vector<float>& query_buffer,
            std::size_t query_count,
            const RerankExactParams& rerank_params,
            RerankBatchProfile* rerank_profile) {
            return RerankExact::RunBatchFloat32(resources, topology_results, query_buffer,
                                               query_count, rerank_params, rerank_profile);
        },
        "TopoAnnsSearch::RunBatchFloat32");
}

TopoAnnsBatchResult TopoAnnsSearch::RunBatchFloat32(
    const SearchResources& resources,
    const DeviceEntryBatch& entry_batch,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const TopoAnnsSearchParams& params) {
    if (entry_batch.num_queries != num_queries) {
        throw std::runtime_error(BuildErrorMessage(
            "TopoAnnsSearch::RunBatchFloat32", "entry batch size must equal num_queries."));
    }
    return RunBatchImpl(
        resources, queries, num_queries, params,
        [&]() { return PqDistanceOracle::FromFloatQueries(resources, queries, num_queries); },
        [&](const PqDistanceOracle& oracle,
            const TopoAnnsSearchParams& effective_params,
            TopologySearchBatchProfile* topology_profile) {
            return TopologySearch::RunBatch(resources, entry_batch, oracle,
                                            effective_params.topology, topology_profile);
        },
        [&](const std::vector<TopologySearchResult>& topology_results,
            const std::vector<float>& query_buffer,
            std::size_t query_count,
            const RerankExactParams& rerank_params,
            RerankBatchProfile* rerank_profile) {
            return RerankExact::RunBatchFloat32(resources, topology_results, query_buffer,
                                               query_count, rerank_params, rerank_profile);
        },
        "TopoAnnsSearch::RunBatchFloat32");
}

TopoAnnsBatchResult TopoAnnsSearch::RunBatchFloat32Fused(
    const SearchResources& resources,
    const DeviceEntryBatch& entry_batch,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const TopoAnnsSearchParams& params) {
    const FusedTopologyArtifacts artifacts = BuildFusedTopologyArtifacts(
        resources, entry_batch, queries, num_queries, params.topology, params.rerank.top_n);
    RerankBatchProfile rerank_profile;
    const RerankBatchResult rerank_results = detail::RunBatchFloat32FromDeviceTopology(
        resources, artifacts.topology_device, queries, num_queries, params.rerank, &rerank_profile);

    return BuildFusedBatchResult(num_queries, artifacts.pq_profile, artifacts.topology_profile,
                                 artifacts.host_stats, rerank_results, rerank_profile);
}

std::vector<TopoAnnsBatchResult> TopoAnnsSearch::RunBatchFloat32FusedRerankSweep(
    const SearchResources& resources,
    const DeviceEntryBatch& entry_batch,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const TopologySearchParams& topology_params,
    const std::vector<RerankExactParams>& rerank_params_list) {
    if (rerank_params_list.empty()) {
        return {};
    }
    std::size_t max_top_n = 0;
    for (const auto& rerank_params : rerank_params_list) {
        max_top_n = std::max(max_top_n, rerank_params.top_n);
    }
    const FusedTopologyArtifacts artifacts = BuildFusedTopologyArtifacts(
        resources, entry_batch, queries, num_queries, topology_params, max_top_n);

    std::vector<TopoAnnsBatchResult> results;
    results.reserve(rerank_params_list.size());
    for (const auto& rerank_params : rerank_params_list) {
        RerankBatchProfile rerank_profile;
        RerankBatchResult rerank_results = detail::RunBatchFloat32FromDeviceTopology(
            resources, artifacts.topology_device, queries, num_queries, rerank_params,
            &rerank_profile);
        results.push_back(BuildFusedBatchResult(
            num_queries, artifacts.pq_profile, artifacts.topology_profile, artifacts.host_stats,
            std::move(rerank_results), rerank_profile));
    }
    return results;
}

TopoAnnsBatchResult TopoAnnsSearch::RunBatchUint8(
    const SearchResources& resources,
    const EntryProvider& entry_provider,
    const std::vector<std::uint8_t>& queries,
    std::size_t num_queries,
    const TopoAnnsSearchParams& params) {
    return RunBatchImpl(
        resources, queries, num_queries, params,
        [&]() { return PqDistanceOracle::FromUint8Queries(resources, queries, num_queries); },
        [&](const PqDistanceOracle& oracle,
            const TopoAnnsSearchParams& effective_params,
            TopologySearchBatchProfile* topology_profile) {
            return TopologySearch::RunBatch(resources, entry_provider, oracle,
                                            effective_params.topology, topology_profile);
        },
        [&](const std::vector<TopologySearchResult>& topology_results,
            const std::vector<std::uint8_t>& query_buffer,
            std::size_t query_count,
            const RerankExactParams& rerank_params,
            RerankBatchProfile* rerank_profile) {
            return RerankExact::RunBatchUint8(resources, topology_results, query_buffer,
                                             query_count, rerank_params, rerank_profile);
        },
        "TopoAnnsSearch::RunBatchUint8");
}

TopoAnnsBatchResult TopoAnnsSearch::RunBatchInt8(
    const SearchResources& resources,
    const EntryProvider& entry_provider,
    const std::vector<std::int8_t>& queries,
    std::size_t num_queries,
    const TopoAnnsSearchParams& params) {
    return RunBatchImpl(
        resources, queries, num_queries, params,
        [&]() { return PqDistanceOracle::FromInt8Queries(resources, queries, num_queries); },
        [&](const PqDistanceOracle& oracle,
            const TopoAnnsSearchParams& effective_params,
            TopologySearchBatchProfile* topology_profile) {
            return TopologySearch::RunBatch(resources, entry_provider, oracle,
                                            effective_params.topology, topology_profile);
        },
        [&](const std::vector<TopologySearchResult>& topology_results,
            const std::vector<std::int8_t>& query_buffer,
            std::size_t query_count,
            const RerankExactParams& rerank_params,
            RerankBatchProfile* rerank_profile) {
            return RerankExact::RunBatchInt8(resources, topology_results, query_buffer,
                                            query_count, rerank_params, rerank_profile);
        },
        "TopoAnnsSearch::RunBatchInt8");
}

}  // namespace topoanns
