#include "topoanns/pq_distance_oracle.hpp"

#include <stdexcept>
#include <utility>

namespace topoanns {

PqDistanceOracle::PqDistanceOracle(const PqIndex& pq_index,
                                   PqQueryDistanceTables query_tables)
    : pq_index_(&pq_index), query_tables_(std::move(query_tables)) {}

PqDistanceOracle PqDistanceOracle::FromFloatQueries(const PqIndex& pq_index,
                                                    const std::vector<float>& queries,
                                                    std::size_t num_queries) {
    return PqDistanceOracle(
        pq_index, PqQueryDistanceTables::FromFloatQueries(pq_index, queries, num_queries));
}

PqDistanceOracle PqDistanceOracle::FromFloatQueriesDeviceOnly(
    const PqIndex& pq_index,
    const std::vector<float>& queries,
    std::size_t num_queries) {
    return PqDistanceOracle(pq_index, PqQueryDistanceTables::FromFloatQueriesDeviceOnly(
                                          pq_index, queries, num_queries));
}

PqDistanceOracle PqDistanceOracle::FromFloatQueries(const SearchResources& resources,
                                                    const std::vector<float>& queries,
                                                    std::size_t num_queries) {
    return FromFloatQueries(resources.pq_index(), queries, num_queries);
}

PqDistanceOracle PqDistanceOracle::FromFloatQueriesDeviceOnly(
    const SearchResources& resources,
    const std::vector<float>& queries,
    std::size_t num_queries) {
    return FromFloatQueriesDeviceOnly(resources.pq_index(), queries, num_queries);
}

PqDistanceOracle PqDistanceOracle::FromUint8Queries(
    const PqIndex& pq_index,
    const std::vector<std::uint8_t>& queries,
    std::size_t num_queries) {
    return PqDistanceOracle(
        pq_index, PqQueryDistanceTables::FromUint8Queries(pq_index, queries, num_queries));
}

PqDistanceOracle PqDistanceOracle::FromUint8Queries(
    const SearchResources& resources,
    const std::vector<std::uint8_t>& queries,
    std::size_t num_queries) {
    return FromUint8Queries(resources.pq_index(), queries, num_queries);
}

PqDistanceOracle PqDistanceOracle::FromInt8Queries(
    const PqIndex& pq_index,
    const std::vector<std::int8_t>& queries,
    std::size_t num_queries) {
    return PqDistanceOracle(
        pq_index, PqQueryDistanceTables::FromInt8Queries(pq_index, queries, num_queries));
}

PqDistanceOracle PqDistanceOracle::FromInt8Queries(
    const SearchResources& resources,
    const std::vector<std::int8_t>& queries,
    std::size_t num_queries) {
    return FromInt8Queries(resources.pq_index(), queries, num_queries);
}

float PqDistanceOracle::Distance(std::size_t query_id, std::uint32_t node_id) const {
    if (pq_index_ == nullptr) {
        throw std::runtime_error(
            BuildErrorMessage("PqDistanceOracle::Distance", "PQ index is not set."));
    }
    return query_tables_.Distance(query_id, node_id, *pq_index_);
}

const PqIndex& PqDistanceOracle::pq_index() const {
    if (pq_index_ == nullptr) {
        throw std::runtime_error(
            BuildErrorMessage("PqDistanceOracle::pq_index", "PQ index is not set."));
    }
    return *pq_index_;
}

}  // namespace topoanns
