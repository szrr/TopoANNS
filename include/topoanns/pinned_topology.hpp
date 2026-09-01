#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "topoanns/topology_layout.hpp"

namespace topoanns {

class PinnedTopology {
public:
    PinnedTopology() = default;
    ~PinnedTopology();

    PinnedTopology(const PinnedTopology&) = delete;
    PinnedTopology& operator=(const PinnedTopology&) = delete;

    PinnedTopology(PinnedTopology&& other) noexcept;
    PinnedTopology& operator=(PinnedTopology&& other) noexcept;

    static PinnedTopology FromDataset(const TopologyDataset& dataset);
    static PinnedTopology FromFile(const std::filesystem::path& path);
    static PinnedTopology FromRaw(std::uint64_t num_nodes,
                                  std::uint32_t degree,
                                  const std::vector<std::uint32_t>& adjacency);

    std::uint64_t num_nodes() const { return num_nodes_; }
    std::uint32_t degree() const { return degree_; }
    const std::uint32_t* host_data() const { return host_data_; }
    const std::uint32_t* device_data() const { return device_data_; }

    std::vector<std::uint32_t> ReadHostNode(std::uint64_t node_id) const;
    std::vector<std::uint32_t> CopyNodeFromDevice(std::uint64_t node_id) const;

private:
    void Release() noexcept;

    std::uint32_t* host_data_ = nullptr;
    std::uint32_t* device_data_ = nullptr;
    std::uint64_t num_nodes_ = 0;
    std::uint32_t degree_ = 0;
    std::size_t element_count_ = 0;
};

}  // namespace topoanns
