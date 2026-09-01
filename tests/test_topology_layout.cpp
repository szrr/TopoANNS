#include "topoanns/topology_layout.hpp"

#include <filesystem>
#include <iostream>
#include <vector>

#include "test_support.hpp"

int main() {
    using topoanns::TopologyDataset;
    using topoanns::TopologyLayout;
    using topoanns::kInvalidNodeId;

    const std::vector<std::vector<std::uint32_t>> adjacency_lists = {
        {1, 2, 3},
        {0, 2},
        {1},
    };

    const TopologyDataset dataset =
        TopologyLayout::MakePadded(3, adjacency_lists);

    TOPOANNS_ASSERT_EQ(dataset.num_nodes, 3ULL);
    TOPOANNS_ASSERT_EQ(dataset.degree, 64U);
    TOPOANNS_ASSERT_EQ(dataset.adjacency.size(), 3ULL * 64ULL);
    TOPOANNS_ASSERT_EQ(TopologyLayout::NodeBytes(), 256ULL);
    TOPOANNS_ASSERT_EQ(TopologyLayout::NodeOffsetBytes(0), 4096ULL);
    TOPOANNS_ASSERT_EQ(TopologyLayout::NodeOffsetBytes(2), 4096ULL + 512ULL);

    const std::uint32_t* node0 = dataset.NodeNeighbors(0);
    TOPOANNS_ASSERT_EQ(node0[0], 1U);
    TOPOANNS_ASSERT_EQ(node0[1], 2U);
    TOPOANNS_ASSERT_EQ(node0[2], 3U);
    TOPOANNS_ASSERT_EQ(node0[3], kInvalidNodeId);

    const std::filesystem::path out_path =
        std::filesystem::current_path() / "test_topology_layout.bin";
    TopologyLayout::WriteFile(out_path, dataset);

    const TopologyDataset loaded = TopologyLayout::ReadFile(out_path);
    TOPOANNS_ASSERT_EQ(loaded.num_nodes, dataset.num_nodes);
    TOPOANNS_ASSERT_EQ(loaded.degree, dataset.degree);
    TOPOANNS_ASSERT_EQ(loaded.adjacency.size(), dataset.adjacency.size());

    for (std::size_t i = 0; i < dataset.adjacency.size(); ++i) {
        TOPOANNS_ASSERT_EQ(loaded.adjacency[i], dataset.adjacency[i]);
    }

    std::cout << "test_topology_layout passed" << std::endl;
    return 0;
}
