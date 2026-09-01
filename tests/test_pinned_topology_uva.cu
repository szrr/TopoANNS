#include "topoanns/pinned_topology.hpp"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "test_support.hpp"

int main() {
    using topoanns::PinnedTopology;
    using topoanns::TopologyDataset;
    using topoanns::TopologyLayout;
    using topoanns::kInvalidNodeId;

    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        std::cerr << "No CUDA device available for test_pinned_topology_uva" << std::endl;
        return 1;
    }

    const std::vector<std::vector<std::uint32_t>> adjacency_lists = {
        {10, 11, 12},
        {20, 21},
        {30},
    };
    const TopologyDataset dataset =
        TopologyLayout::MakePadded(3, adjacency_lists);
    PinnedTopology pinned = PinnedTopology::FromDataset(dataset);

    TOPOANNS_ASSERT_TRUE(pinned.host_data() != nullptr);
    TOPOANNS_ASSERT_TRUE(pinned.device_data() != nullptr);
    TOPOANNS_ASSERT_EQ(pinned.num_nodes(), 3ULL);
    TOPOANNS_ASSERT_EQ(pinned.degree(), 64U);

    const std::vector<std::uint32_t> host_neighbors = pinned.ReadHostNode(1);
    const std::vector<std::uint32_t> device_neighbors = pinned.CopyNodeFromDevice(1);
    TOPOANNS_ASSERT_EQ(host_neighbors.size(), 64ULL);
    TOPOANNS_ASSERT_EQ(device_neighbors.size(), 64ULL);

    for (std::size_t i = 0; i < host_neighbors.size(); ++i) {
        TOPOANNS_ASSERT_EQ(device_neighbors[i], host_neighbors[i]);
    }

    TOPOANNS_ASSERT_EQ(device_neighbors[0], 20U);
    TOPOANNS_ASSERT_EQ(device_neighbors[1], 21U);
    TOPOANNS_ASSERT_EQ(device_neighbors[2], kInvalidNodeId);

    std::cout << "test_pinned_topology_uva passed" << std::endl;
    return 0;
}
