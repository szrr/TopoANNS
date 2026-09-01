#pragma once

#include <cstddef>
#include <cstdint>

#include "topoanns/cuda_buffer.hpp"
#include "topoanns/device_entry_batch.hpp"

namespace topoanns::detail {

DeviceEntryBatch BuildDeviceEntryBatchFromRvqClusters(
    const CudaBuffer<int>& device_clusters,
    std::size_t num_queries,
    std::size_t entry_count,
    std::uint32_t fallback_entry,
    const CudaBuffer<std::uint32_t>& cluster_offsets,
    const CudaBuffer<std::uint32_t>& cluster_ids,
    double* out_gather_ms);

}  // namespace topoanns::detail
