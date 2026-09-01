#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "topoanns/common.hpp"
#include "topoanns/cuda_buffer.hpp"
#include "topoanns/pq_index.hpp"

namespace topoanns {

struct PqQueryTablesProfile {
    double total_ms = 0.0;
    double query_upload_ms = 0.0;
    double zero_fill_ms = 0.0;
    double kernel_ms = 0.0;
    double table_download_ms = 0.0;
};

class PqQueryDistanceTables {
public:
    PqQueryDistanceTables() = default;
    PqQueryDistanceTables(PqQueryDistanceTables&&) noexcept = default;
    PqQueryDistanceTables& operator=(PqQueryDistanceTables&&) noexcept = default;

    PqQueryDistanceTables(const PqQueryDistanceTables&) = delete;
    PqQueryDistanceTables& operator=(const PqQueryDistanceTables&) = delete;

    static PqQueryDistanceTables FromFloatQueries(const PqIndex& pq_index,
                                                  const std::vector<float>& queries,
                                                  std::size_t num_queries);
    static PqQueryDistanceTables FromFloatQueriesDeviceOnly(const PqIndex& pq_index,
                                                            const std::vector<float>& queries,
                                                            std::size_t num_queries);
    static PqQueryDistanceTables FromFloatQueriesDeviceBufferAsync(
        const PqIndex& pq_index,
        const CudaBuffer<float>& device_queries,
        std::size_t num_queries,
        cudaStream_t stream);
    static PqQueryDistanceTables FromUint8Queries(const PqIndex& pq_index,
                                                  const std::vector<std::uint8_t>& queries,
                                                  std::size_t num_queries);
    static PqQueryDistanceTables FromInt8Queries(const PqIndex& pq_index,
                                                 const std::vector<std::int8_t>& queries,
                                                 std::size_t num_queries);

    std::size_t num_queries() const { return num_queries_; }
    std::size_t num_chunks() const { return num_chunks_; }
    bool has_host_tables() const { return host_tables_materialized_; }
    const std::vector<float>& host_tables() const { return host_tables_; }
    const CudaBuffer<float>& device_tables() const { return device_tables_; }
    const PqQueryTablesProfile& profile() const { return profile_; }

    float Lookup(std::size_t query_id, std::size_t chunk_id,
                 std::uint32_t centroid_id) const;
    float Distance(std::size_t query_id, std::uint32_t node_id,
                   const PqIndex& pq_index) const;

private:
    template <typename T>
    static PqQueryDistanceTables BuildFromQueries(const PqIndex& pq_index,
                                                  const std::vector<T>& queries,
                                                  std::size_t num_queries,
                                                  bool materialize_host_tables,
                                                  const char* context);

    std::size_t num_queries_ = 0;
    std::size_t num_chunks_ = 0;
    bool host_tables_materialized_ = false;
    std::vector<float> host_tables_;
    CudaBuffer<float> device_tables_;
    PqQueryTablesProfile profile_;
};

}  // namespace topoanns
