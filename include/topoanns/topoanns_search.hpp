#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "topoanns/device_entry_batch.hpp"
#include "topoanns/entry_provider.hpp"
#include "topoanns/pq_query_tables.hpp"
#include "topoanns/rerank_exact.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/topology_search.hpp"

namespace topoanns {

struct TopoAnnsSearchParams {
    TopologySearchParams topology;
    RerankExactParams rerank;
};

struct TopoAnnsQueryResult {
    TopologySearchResult topology;
    RerankQueryResult rerank;
};

struct TopoAnnsBatchResult {
    std::vector<TopoAnnsQueryResult> queries;
    RerankBatchStats rerank_stats;
    PqQueryTablesProfile pq_profile;
    TopologySearchBatchProfile topology_profile;
    RerankBatchProfile rerank_profile;
};

class TopoAnnsSearch final {
public:
    static TopoAnnsBatchResult RunBatchFloat32(
        const SearchResources& resources,
        const EntryProvider& entry_provider,
        const std::vector<float>& queries,
        std::size_t num_queries,
        const TopoAnnsSearchParams& params);

    static TopoAnnsBatchResult RunBatchFloat32(
        const SearchResources& resources,
        const DeviceEntryBatch& entry_batch,
        const std::vector<float>& queries,
        std::size_t num_queries,
        const TopoAnnsSearchParams& params);

    static TopoAnnsBatchResult RunBatchFloat32Fused(
        const SearchResources& resources,
        const DeviceEntryBatch& entry_batch,
        const std::vector<float>& queries,
        std::size_t num_queries,
        const TopoAnnsSearchParams& params);

    static std::vector<TopoAnnsBatchResult> RunBatchFloat32FusedRerankSweep(
        const SearchResources& resources,
        const DeviceEntryBatch& entry_batch,
        const std::vector<float>& queries,
        std::size_t num_queries,
        const TopologySearchParams& topology_params,
        const std::vector<RerankExactParams>& rerank_params_list);

    static TopoAnnsBatchResult RunBatchUint8(
        const SearchResources& resources,
        const EntryProvider& entry_provider,
        const std::vector<std::uint8_t>& queries,
        std::size_t num_queries,
        const TopoAnnsSearchParams& params);

    static TopoAnnsBatchResult RunBatchInt8(
        const SearchResources& resources,
        const EntryProvider& entry_provider,
        const std::vector<std::int8_t>& queries,
        std::size_t num_queries,
        const TopoAnnsSearchParams& params);
};

}  // namespace topoanns
