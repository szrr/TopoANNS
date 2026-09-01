#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "topoanns/pq_query_tables.hpp"
#include "topoanns/search_resources.hpp"

namespace topoanns {

class PqDistanceOracle final {
public:
    PqDistanceOracle(const PqIndex& pq_index, PqQueryDistanceTables query_tables);

    static PqDistanceOracle FromFloatQueries(const PqIndex& pq_index,
                                             const std::vector<float>& queries,
                                             std::size_t num_queries);
    static PqDistanceOracle FromFloatQueriesDeviceOnly(const PqIndex& pq_index,
                                                       const std::vector<float>& queries,
                                                       std::size_t num_queries);
    static PqDistanceOracle FromFloatQueries(const SearchResources& resources,
                                             const std::vector<float>& queries,
                                             std::size_t num_queries);
    static PqDistanceOracle FromFloatQueriesDeviceOnly(const SearchResources& resources,
                                                       const std::vector<float>& queries,
                                                       std::size_t num_queries);
    static PqDistanceOracle FromUint8Queries(const PqIndex& pq_index,
                                             const std::vector<std::uint8_t>& queries,
                                             std::size_t num_queries);
    static PqDistanceOracle FromUint8Queries(const SearchResources& resources,
                                             const std::vector<std::uint8_t>& queries,
                                             std::size_t num_queries);
    static PqDistanceOracle FromInt8Queries(const PqIndex& pq_index,
                                            const std::vector<std::int8_t>& queries,
                                            std::size_t num_queries);
    static PqDistanceOracle FromInt8Queries(const SearchResources& resources,
                                            const std::vector<std::int8_t>& queries,
                                            std::size_t num_queries);

    float Distance(std::size_t query_id, std::uint32_t node_id) const;

    std::size_t num_queries() const { return query_tables_.num_queries(); }
    const PqIndex& pq_index() const;
    const PqQueryDistanceTables& query_tables() const { return query_tables_; }

private:
    const PqIndex* pq_index_ = nullptr;
    PqQueryDistanceTables query_tables_;
};

}  // namespace topoanns
