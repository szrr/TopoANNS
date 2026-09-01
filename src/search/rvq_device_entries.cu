#include "rvq_device_entries.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "topoanns/common.hpp"

namespace topoanns::detail {
namespace {

constexpr std::size_t kThreadsPerBlock = 128;

__global__ void gather_rvq_entries_kernel(const int* cluster_ids_by_query,
                                          const std::uint32_t* cluster_offsets,
                                          const std::uint32_t* flat_cluster_ids,
                                          std::size_t entry_count,
                                          std::uint32_t fallback_entry,
                                          std::uint32_t* out_entry_ids) {
    const std::size_t query_id = blockIdx.x;
    const std::size_t entry_base = query_id * entry_count;
    for (std::size_t i = threadIdx.x; i < entry_count; i += blockDim.x) {
        out_entry_ids[entry_base + i] = kInvalidNodeId;
    }
    __syncthreads();

    const int cluster_id = cluster_ids_by_query[query_id];
    if (cluster_id < 0) {
        if (threadIdx.x == 0 && entry_count > 0) {
            out_entry_ids[entry_base] = fallback_entry;
        }
        return;
    }

    const std::uint32_t cluster_begin = cluster_offsets[cluster_id];
    const std::uint32_t cluster_end = cluster_offsets[cluster_id + 1];
    const std::size_t cluster_size =
        static_cast<std::size_t>(cluster_end) - static_cast<std::size_t>(cluster_begin);
    const std::size_t count = cluster_size < entry_count ? cluster_size : entry_count;
    if (count == 0) {
        if (threadIdx.x == 0 && entry_count > 0) {
            out_entry_ids[entry_base] = fallback_entry;
        }
        return;
    }

    for (std::size_t i = threadIdx.x; i < count; i += blockDim.x) {
        out_entry_ids[entry_base + i] =
            flat_cluster_ids[cluster_begin + static_cast<std::uint32_t>(i)];
    }
}

}  // namespace

DeviceEntryBatch BuildDeviceEntryBatchFromRvqClusters(
    const CudaBuffer<int>& device_clusters,
    std::size_t num_queries,
    std::size_t entry_count,
    std::uint32_t fallback_entry,
    const CudaBuffer<std::uint32_t>& cluster_offsets,
    const CudaBuffer<std::uint32_t>& cluster_ids,
    double* out_gather_ms) {
    DeviceEntryBatch batch;
    batch.num_queries = num_queries;
    batch.entries_per_query = entry_count;
    if (num_queries == 0 || entry_count == 0) {
        if (out_gather_ms != nullptr) {
            *out_gather_ms = 0.0;
        }
        return batch;
    }

    std::vector<std::uint32_t> host_offsets(num_queries + 1, 0);
    for (std::size_t query_id = 0; query_id <= num_queries; ++query_id) {
        host_offsets[query_id] = static_cast<std::uint32_t>(query_id * entry_count);
    }
    batch.offsets = CudaBuffer<std::uint32_t>::CopyFromHost(host_offsets);
    batch.ids = CudaBuffer<std::uint32_t>::Allocate(num_queries * entry_count);

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(begin), "cudaEventRecord");
    gather_rvq_entries_kernel<<<num_queries, kThreadsPerBlock>>>(
        device_clusters.get(), cluster_offsets.get(), cluster_ids.get(), entry_count,
        fallback_entry, batch.ids.get());
    ThrowIfCudaError(cudaGetLastError(), "gather_rvq_entries_kernel");
    ThrowIfCudaError(cudaEventRecord(end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");
    float gather_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&gather_ms, begin, end), "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    if (out_gather_ms != nullptr) {
        *out_gather_ms = static_cast<double>(gather_ms);
    }
    return batch;
}

}  // namespace topoanns::detail
