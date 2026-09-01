#include "topoanns/pq_query_tables.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <stdexcept>
#include <vector>

namespace topoanns {
namespace {

template <typename T>
__global__ void populate_chunk_distances_kernel(
    std::size_t num_queries,
    std::size_t dim,
    std::size_t num_chunks,
    const T* queries,
    float* chunk_tables,
    const float* centroid,
    const std::uint32_t* chunk_offsets,
    const float* pq_tables_col_major) {
    extern __shared__ float shared_query[];

    const std::size_t lane = threadIdx.x;
    const std::size_t query_id = blockIdx.x;
    if (query_id >= num_queries) {
        return;
    }

    const T* query_ptr = queries + query_id * dim;
    float* out_ptr = chunk_tables + query_id * num_chunks * kNumPqCentroids;

    for (std::size_t i = lane; i < dim; i += blockDim.x) {
        shared_query[i] = static_cast<float>(query_ptr[i]) - centroid[i];
    }
    __syncthreads();

    for (std::size_t chunk = 0; chunk < num_chunks; ++chunk) {
        float* chunk_out = out_ptr + chunk * kNumPqCentroids;
        for (std::size_t dim_idx = chunk_offsets[chunk]; dim_idx < chunk_offsets[chunk + 1];
             ++dim_idx) {
            const float* centers_for_dim =
                pq_tables_col_major + dim_idx * kNumPqCentroids;
            const float query_value = shared_query[dim_idx];
            for (std::size_t center = lane; center < kNumPqCentroids; center += blockDim.x) {
                const float diff = centers_for_dim[center] - query_value;
                chunk_out[center] += diff * diff;
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

}  // namespace

template <typename T>
PqQueryDistanceTables PqQueryDistanceTables::BuildFromQueries(
    const PqIndex& pq_index,
    const std::vector<T>& queries,
    std::size_t num_queries,
    bool materialize_host_tables,
    const char* context) {
    if (pq_index.host().ndims == 0 || pq_index.host().num_chunks == 0) {
        throw std::runtime_error(BuildErrorMessage(context, "PQ index is empty."));
    }
    const std::size_t dim = pq_index.host().ndims;
    if (queries.size() != num_queries * dim) {
        throw std::runtime_error(BuildErrorMessage(
            context, "Query buffer size must equal num_queries * ndims."));
    }

    PqQueryDistanceTables tables;
    const auto total_begin = std::chrono::steady_clock::now();
    tables.device_tables_ =
        CudaBuffer<float>::Allocate(num_queries * pq_index.host().num_chunks * kNumPqCentroids);
    const auto zero_begin = std::chrono::steady_clock::now();
    ThrowIfCudaError(cudaMemset(tables.device_tables_.get(), 0,
                                tables.device_tables_.size() * sizeof(float)),
                     "cudaMemset");
    const auto zero_end = std::chrono::steady_clock::now();

    const auto upload_begin = std::chrono::steady_clock::now();
    CudaBuffer<T> query_buffer = CudaBuffer<T>::CopyFromHost(queries);
    const auto upload_end = std::chrono::steady_clock::now();

    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin), "cudaEventRecord");
    populate_chunk_distances_kernel<<<num_queries, 32, dim * sizeof(float)>>>(
        num_queries, dim, pq_index.host().num_chunks, query_buffer.get(),
        tables.device_tables_.get(), pq_index.device().centroid.get(),
        pq_index.device().chunk_offsets.get(), pq_index.device().tables_col_major.get());
    ThrowIfCudaError(cudaGetLastError(), "populate_chunk_distances_kernel");
    ThrowIfCudaError(cudaEventRecord(kernel_end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");

    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);

    tables.num_queries_ = num_queries;
    tables.num_chunks_ = pq_index.host().num_chunks;
    tables.host_tables_materialized_ = materialize_host_tables;
    const auto download_begin = std::chrono::steady_clock::now();
    if (materialize_host_tables) {
        tables.host_tables_ = tables.device_tables_.CopyToHost();
    }
    const auto download_end = std::chrono::steady_clock::now();
    const auto total_end = std::chrono::steady_clock::now();
    tables.profile_.query_upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
    tables.profile_.zero_fill_ms =
        std::chrono::duration<double, std::milli>(zero_end - zero_begin).count();
    tables.profile_.kernel_ms = static_cast<double>(kernel_ms);
    tables.profile_.table_download_ms =
        std::chrono::duration<double, std::milli>(download_end - download_begin).count();
    tables.profile_.total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_begin).count();
    return tables;
}

PqQueryDistanceTables PqQueryDistanceTables::FromFloatQueriesDeviceBufferAsync(
    const PqIndex& pq_index,
    const CudaBuffer<float>& device_queries,
    std::size_t num_queries,
    cudaStream_t stream) {
    if (pq_index.host().ndims == 0 || pq_index.host().num_chunks == 0) {
        throw std::runtime_error(
            BuildErrorMessage("FromFloatQueriesDeviceBufferAsync", "PQ index is empty."));
    }
    const std::size_t dim = pq_index.host().ndims;
    if (device_queries.size() != num_queries * dim) {
        throw std::runtime_error(BuildErrorMessage(
            "FromFloatQueriesDeviceBufferAsync",
            "Device query buffer size must equal num_queries * ndims."));
    }

    PqQueryDistanceTables tables;
    tables.device_tables_ =
        CudaBuffer<float>::Allocate(num_queries * pq_index.host().num_chunks * kNumPqCentroids);
    ThrowIfCudaError(cudaMemsetAsync(tables.device_tables_.get(), 0,
                                     tables.device_tables_.size() * sizeof(float), stream),
                     "cudaMemsetAsync");
    populate_chunk_distances_kernel<<<num_queries, 32, dim * sizeof(float), stream>>>(
        num_queries, dim, pq_index.host().num_chunks, device_queries.get(),
        tables.device_tables_.get(), pq_index.device().centroid.get(),
        pq_index.device().chunk_offsets.get(), pq_index.device().tables_col_major.get());
    ThrowIfCudaError(cudaGetLastError(), "populate_chunk_distances_kernel");
    tables.num_queries_ = num_queries;
    tables.num_chunks_ = pq_index.host().num_chunks;
    tables.host_tables_materialized_ = false;
    return tables;
}

PqQueryDistanceTables PqQueryDistanceTables::FromFloatQueries(
    const PqIndex& pq_index,
    const std::vector<float>& queries,
    std::size_t num_queries) {
    return BuildFromQueries(pq_index, queries, num_queries, true, "FromFloatQueries");
}

PqQueryDistanceTables PqQueryDistanceTables::FromFloatQueriesDeviceOnly(
    const PqIndex& pq_index,
    const std::vector<float>& queries,
    std::size_t num_queries) {
    return BuildFromQueries(pq_index, queries, num_queries, false,
                            "FromFloatQueriesDeviceOnly");
}

PqQueryDistanceTables PqQueryDistanceTables::FromUint8Queries(
    const PqIndex& pq_index,
    const std::vector<std::uint8_t>& queries,
    std::size_t num_queries) {
    return BuildFromQueries(pq_index, queries, num_queries, true, "FromUint8Queries");
}

PqQueryDistanceTables PqQueryDistanceTables::FromInt8Queries(
    const PqIndex& pq_index,
    const std::vector<std::int8_t>& queries,
    std::size_t num_queries) {
    return BuildFromQueries(pq_index, queries, num_queries, true, "FromInt8Queries");
}

float PqQueryDistanceTables::Lookup(std::size_t query_id, std::size_t chunk_id,
                                    std::uint32_t centroid_id) const {
    if (!host_tables_materialized_) {
        throw std::runtime_error(BuildErrorMessage(
            "PqQueryDistanceTables::Lookup",
            "Host query-distance tables are not materialized for this instance."));
    }
    if (query_id >= num_queries_ || chunk_id >= num_chunks_ ||
        centroid_id >= kNumPqCentroids) {
        throw std::runtime_error(
            BuildErrorMessage("PqQueryDistanceTables::Lookup", "index out of range."));
    }
    const std::size_t offset =
        query_id * num_chunks_ * kNumPqCentroids + chunk_id * kNumPqCentroids + centroid_id;
    return host_tables_[offset];
}

float PqQueryDistanceTables::Distance(std::size_t query_id, std::uint32_t node_id,
                                      const PqIndex& pq_index) const {
    if (!host_tables_materialized_) {
        throw std::runtime_error(BuildErrorMessage(
            "PqQueryDistanceTables::Distance",
            "Host query-distance tables are not materialized for this instance."));
    }
    if (query_id >= num_queries_ || node_id >= pq_index.host().num_points) {
        throw std::runtime_error(
            BuildErrorMessage("PqQueryDistanceTables::Distance", "index out of range."));
    }
    float distance = 0.0f;
    std::vector<std::uint8_t> device_codes;
    const std::uint8_t* host_codes = nullptr;
    if (pq_index.host().codes_on_host) {
        host_codes = pq_index.host_codes() + node_id * num_chunks_;
    } else {
        device_codes.resize(num_chunks_);
        ThrowIfCudaError(
            cudaMemcpy(device_codes.data(),
                       pq_index.device_codes() + node_id * num_chunks_,
                       num_chunks_ * sizeof(std::uint8_t),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpyDeviceToHost");
        host_codes = device_codes.data();
    }
    for (std::size_t chunk = 0; chunk < num_chunks_; ++chunk) {
        const std::uint8_t center_id = host_codes[chunk];
        distance += Lookup(query_id, chunk, center_id);
    }
    return distance;
}

}  // namespace topoanns
