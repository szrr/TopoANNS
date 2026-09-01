#include "topoanns/topology_search.hpp"

#include <stdexcept>
#include <vector>

#include "topology_search_kernel.hpp"

namespace topoanns {

TopologySearchResult TopologySearch::Run(const SearchResources& resources,
                                         const EntryProvider& entry_provider,
                                         const PqDistanceOracle& distance_oracle,
                                         std::size_t query_id,
                                         const TopologySearchParams& params,
                                         TopologySearchBatchProfile* out_profile) {
    if (params.top_k == 0) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::Run", "top_k must be positive."));
    }
    if (params.top_l == 0) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::Run", "top_l must be positive."));
    }
    if (params.candidate_queue_size != 0 && params.candidate_queue_size < params.top_l) {
        throw std::runtime_error(BuildErrorMessage(
            "TopologySearch::Run", "candidate_queue_size must be zero or at least top_l."));
    }
    if (params.search_width == 0) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::Run", "search_width must be positive."));
    }
    if (query_id >= distance_oracle.num_queries()) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::Run", "query_id is out of range."));
    }

    const std::vector<std::uint32_t> entries = entry_provider.GetEntryPoints(query_id);
    std::vector<std::vector<std::uint32_t>> entries_by_query(1);
    entries_by_query[0] = entries;
    return detail::RunTopologySearchKernelBatch(resources, distance_oracle, entries_by_query,
                                                query_id, 1, params, out_profile)
        .front();
}

std::vector<TopologySearchResult> TopologySearch::RunBatch(
    const SearchResources& resources,
    const EntryProvider& entry_provider,
    const PqDistanceOracle& distance_oracle,
    const TopologySearchParams& params,
    TopologySearchBatchProfile* out_profile) {
    if (params.top_k == 0) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::RunBatch", "top_k must be positive."));
    }
    if (params.top_l == 0) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::RunBatch", "top_l must be positive."));
    }
    if (params.candidate_queue_size != 0 && params.candidate_queue_size < params.top_l) {
        throw std::runtime_error(BuildErrorMessage(
            "TopologySearch::RunBatch",
            "candidate_queue_size must be zero or at least top_l."));
    }
    if (params.search_width == 0) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::RunBatch", "search_width must be positive."));
    }

    std::vector<std::vector<std::uint32_t>> entries_by_query(distance_oracle.num_queries());
    for (std::size_t query_id = 0; query_id < distance_oracle.num_queries(); ++query_id) {
        entries_by_query[query_id] = entry_provider.GetEntryPoints(query_id);
    }
    return detail::RunTopologySearchKernelBatch(resources, distance_oracle, entries_by_query, 0,
                                                distance_oracle.num_queries(), params,
                                                out_profile);
}

std::vector<TopologySearchResult> TopologySearch::RunBatch(
    const SearchResources& resources,
    const DeviceEntryBatch& entry_batch,
    const PqDistanceOracle& distance_oracle,
    const TopologySearchParams& params,
    TopologySearchBatchProfile* out_profile) {
    if (params.top_k == 0) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::RunBatch", "top_k must be positive."));
    }
    if (params.top_l == 0) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::RunBatch", "top_l must be positive."));
    }
    if (params.candidate_queue_size != 0 && params.candidate_queue_size < params.top_l) {
        throw std::runtime_error(BuildErrorMessage(
            "TopologySearch::RunBatch",
            "candidate_queue_size must be zero or at least top_l."));
    }
    if (params.search_width == 0) {
        throw std::runtime_error(
            BuildErrorMessage("TopologySearch::RunBatch", "search_width must be positive."));
    }
    if (entry_batch.num_queries != distance_oracle.num_queries()) {
        throw std::runtime_error(BuildErrorMessage(
            "TopologySearch::RunBatch", "entry batch size must match oracle query count."));
    }

    return detail::RunTopologySearchKernelBatch(resources, distance_oracle, entry_batch, params,
                                                out_profile);
}

}  // namespace topoanns
