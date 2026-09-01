#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace topoanns::detail {

struct DeviceTopologyIoRequest {
    std::uint32_t global_query_id = 0;
    std::uint32_t response_index = 0;
    std::uint32_t node_id = 0;
    std::uint32_t reserved = 0;
};

struct TopologyMicrobatchIoConfig {
    std::uint64_t num_nodes = 0;
    std::uint32_t degree = 0;
    std::uint32_t topology_nodes_per_page = 0;
    std::uint32_t combined_node_bytes = 0;
    std::uint32_t combined_nodes_per_page = 0;
    std::uint32_t query_dim = 0;
    const float* queries = nullptr;
    const std::uint32_t* validation_topology = nullptr;
    unsigned long long* validation_mismatch_neighbors = nullptr;
};

void LaunchBamTopologyMicrobatchReads(
    const void* device_read_handle,
    const DeviceTopologyIoRequest* requests,
    const std::uint32_t* request_count,
    std::size_t request_capacity,
    const TopologyMicrobatchIoConfig& config,
    std::uint32_t* response_neighbors,
    float* response_exact_distances,
    std::size_t num_blocks,
    std::size_t threads_per_block,
    cudaStream_t stream);

}  // namespace topoanns::detail
