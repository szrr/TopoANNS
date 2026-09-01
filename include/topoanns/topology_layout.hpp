#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "topoanns/common.hpp"

namespace topoanns {

struct TopologyHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t degree;
    std::uint64_t num_nodes;
    std::uint64_t header_bytes;
    std::uint64_t payload_bytes;
    std::uint8_t reserved[kDefaultPageSizeBytes - 40];
};

static_assert(sizeof(TopologyHeader) == kDefaultPageSizeBytes,
              "TopologyHeader must be exactly one 4KB page.");

struct TopologyDataset {
    std::uint64_t num_nodes = 0;
    std::uint32_t degree = kFixedTopologyDegree;
    std::vector<std::uint32_t> adjacency;

    std::size_t NodeBytes() const;
    const std::uint32_t* NodeNeighbors(std::uint64_t node_id) const;
};

class TopologyLayout {
public:
    static constexpr std::uint32_t kVersion = 1;

    static std::size_t NodeBytes();
    static std::uint64_t NodeOffsetBytes(std::uint64_t node_id);
    static TopologyHeader ReadHeader(const std::filesystem::path& path);
    static TopologyDataset MakePadded(
        std::uint64_t num_nodes,
        const std::vector<std::vector<std::uint32_t>>& adjacency_lists);
    static void WriteFile(const std::filesystem::path& path,
                          const TopologyDataset& dataset);
    static TopologyDataset ReadFile(const std::filesystem::path& path);
};

}  // namespace topoanns
