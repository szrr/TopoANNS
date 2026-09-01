#include "topoanns/topology_layout.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace topoanns {
namespace {

constexpr char kTopologyMagic[8] = {'T', 'O', 'P', 'O', 'A', 'N', 'N', 'S'};

void ValidateHeader(const TopologyHeader& header,
                    std::streamsize file_size,
                    const char* context) {
    if (std::memcmp(header.magic, kTopologyMagic, sizeof(header.magic)) != 0) {
        throw std::runtime_error(
            BuildErrorMessage(context, "Unexpected topology magic."));
    }
    if (header.version != TopologyLayout::kVersion) {
        throw std::runtime_error(
            BuildErrorMessage(context, "Unexpected topology version."));
    }
    if (header.degree != kFixedTopologyDegree) {
        throw std::runtime_error(
            BuildErrorMessage(context, "Unexpected topology degree."));
    }

    const std::size_t adjacency_size =
        static_cast<std::size_t>(header.num_nodes) * header.degree;
    const std::streamsize expected_size =
        static_cast<std::streamsize>(sizeof(TopologyHeader) +
                                     adjacency_size * sizeof(std::uint32_t));
    if (file_size != expected_size) {
        throw std::runtime_error(BuildErrorMessage(
            context, "Topology file size does not match metadata."));
    }
}

TopologyHeader BuildHeader(const TopologyDataset& dataset) {
    TopologyHeader header{};
    std::memcpy(header.magic, kTopologyMagic, sizeof(header.magic));
    header.version = TopologyLayout::kVersion;
    header.degree = dataset.degree;
    header.num_nodes = dataset.num_nodes;
    header.header_bytes = sizeof(TopologyHeader);
    header.payload_bytes = dataset.adjacency.size() * sizeof(std::uint32_t);
    return header;
}

void ValidateDataset(const TopologyDataset& dataset) {
    if (dataset.degree != kFixedTopologyDegree) {
        throw std::runtime_error(BuildErrorMessage(
            "ValidateDataset", "Topology degree must be fixed at 64."));
    }
    const std::size_t expected_size =
        static_cast<std::size_t>(dataset.num_nodes) * dataset.degree;
    if (dataset.adjacency.size() != expected_size) {
        throw std::runtime_error(BuildErrorMessage(
            "ValidateDataset", "Adjacency vector size does not match num_nodes * degree."));
    }
}

}  // namespace

std::size_t TopologyDataset::NodeBytes() const {
    return static_cast<std::size_t>(degree) * sizeof(std::uint32_t);
}

const std::uint32_t* TopologyDataset::NodeNeighbors(std::uint64_t node_id) const {
    if (node_id >= num_nodes) {
        throw std::runtime_error(BuildErrorMessage("NodeNeighbors", "node_id is out of range."));
    }
    return adjacency.data() + node_id * degree;
}

std::size_t TopologyLayout::NodeBytes() {
    return static_cast<std::size_t>(kFixedTopologyDegree) * sizeof(std::uint32_t);
}

std::uint64_t TopologyLayout::NodeOffsetBytes(std::uint64_t node_id) {
    return sizeof(TopologyHeader) + node_id * NodeBytes();
}

TopologyDataset TopologyLayout::MakePadded(
    std::uint64_t num_nodes,
    const std::vector<std::vector<std::uint32_t>>& adjacency_lists) {
    if (adjacency_lists.size() != num_nodes) {
        throw std::runtime_error(BuildErrorMessage(
            "MakePadded", "adjacency_lists size must match num_nodes."));
    }

    TopologyDataset dataset;
    dataset.num_nodes = num_nodes;
    dataset.degree = kFixedTopologyDegree;
    dataset.adjacency.assign(
        static_cast<std::size_t>(num_nodes) * dataset.degree, kInvalidNodeId);

    for (std::uint64_t node_id = 0; node_id < num_nodes; ++node_id) {
        const auto& neighbors = adjacency_lists[static_cast<std::size_t>(node_id)];
        if (neighbors.size() > dataset.degree) {
            throw std::runtime_error(BuildErrorMessage(
                "MakePadded", "A node has more than 64 neighbors."));
        }
        std::uint32_t* dst = dataset.adjacency.data() + node_id * dataset.degree;
        for (std::size_t i = 0; i < neighbors.size(); ++i) {
            dst[i] = neighbors[i];
        }
    }

    return dataset;
}

void TopologyLayout::WriteFile(const std::filesystem::path& path,
                               const TopologyDataset& dataset) {
    ValidateDataset(dataset);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("WriteFile", "Failed to open output file."));
    }

    const TopologyHeader header = BuildHeader(dataset);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(dataset.adjacency.data()),
              static_cast<std::streamsize>(dataset.adjacency.size() *
                                           sizeof(std::uint32_t)));
    if (!out.good()) {
        throw std::runtime_error(
            BuildErrorMessage("WriteFile", "Failed while writing topology file."));
    }
}

TopologyDataset TopologyLayout::ReadFile(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadFile", "Failed to open topology file."));
    }

    const std::streamsize file_size = in.tellg();
    if (file_size < static_cast<std::streamsize>(sizeof(TopologyHeader))) {
        throw std::runtime_error(
            BuildErrorMessage("ReadFile", "Topology file is smaller than the header."));
    }

    in.seekg(0, std::ios::beg);
    TopologyHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadFile", "Failed to read topology header."));
    }
    ValidateHeader(header, file_size, "ReadFile");
    const std::size_t adjacency_size =
        static_cast<std::size_t>(header.num_nodes) * header.degree;

    TopologyDataset dataset;
    dataset.num_nodes = header.num_nodes;
    dataset.degree = header.degree;
    dataset.adjacency.resize(adjacency_size);
    in.read(reinterpret_cast<char*>(dataset.adjacency.data()),
            static_cast<std::streamsize>(dataset.adjacency.size() *
                                         sizeof(std::uint32_t)));
    if (!in.good()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadFile", "Failed to read adjacency payload."));
    }

    return dataset;
}

TopologyHeader TopologyLayout::ReadHeader(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadHeader", "Failed to open topology file."));
    }

    const std::streamsize file_size = in.tellg();
    if (file_size < static_cast<std::streamsize>(sizeof(TopologyHeader))) {
        throw std::runtime_error(
            BuildErrorMessage("ReadHeader", "Topology file is smaller than the header."));
    }

    in.seekg(0, std::ios::beg);
    TopologyHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadHeader", "Failed to read topology header."));
    }
    ValidateHeader(header, file_size, "ReadHeader");
    return header;
}

}  // namespace topoanns
