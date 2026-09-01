#include "topoanns/diskann_disk_index.hpp"

#include <fstream>
#include <stdexcept>
#include <vector>

namespace topoanns {

DiskannDiskIndexLayout DiskannDiskIndexLayout::Load(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("DiskannDiskIndexLayout::Load", "Failed to open " + path.string()));
    }

    std::uint32_t meta_count = 0;
    std::uint32_t meta_dim = 0;
    in.read(reinterpret_cast<char*>(&meta_count), sizeof(meta_count));
    in.read(reinterpret_cast<char*>(&meta_dim), sizeof(meta_dim));
    if (!in.good() || meta_dim != 1 || meta_count != 9) {
        throw std::runtime_error(BuildErrorMessage(
            "DiskannDiskIndexLayout::Load", "Unexpected DiskANN metadata header."));
    }

    std::vector<std::uint64_t> meta_values(meta_count, 0);
    in.read(reinterpret_cast<char*>(meta_values.data()),
            static_cast<std::streamsize>(meta_values.size() * sizeof(std::uint64_t)));
    if (!in.good()) {
        throw std::runtime_error(BuildErrorMessage(
            "DiskannDiskIndexLayout::Load", "Failed to read metadata payload."));
    }

    return DiskannDiskIndexLayout(DiskannDiskIndexMetadata{
        meta_values[0],
        meta_values[1],
        meta_values[2],
        meta_values[3],
        meta_values[4],
        meta_values[5],
        meta_values[6],
        meta_values[7],
        meta_values[8],
    });
}

std::size_t DiskannDiskIndexLayout::coord_bytes() const {
    return static_cast<std::size_t>(metadata_.vector_dim) * sizeof(float);
}

std::size_t DiskannDiskIndexLayout::neighbor_capacity() const {
    if (metadata_.max_node_len < coord_bytes() + sizeof(std::uint32_t)) {
        throw std::runtime_error(BuildErrorMessage(
            "DiskannDiskIndexLayout::neighbor_capacity", "Invalid max_node_len."));
    }
    return static_cast<std::size_t>(
        (metadata_.max_node_len - coord_bytes() - sizeof(std::uint32_t)) / sizeof(std::uint32_t));
}

std::uint64_t DiskannDiskIndexLayout::sectors_per_node() const {
    return (metadata_.max_node_len + kSectorBytes - 1) / kSectorBytes;
}

std::uint64_t DiskannDiskIndexLayout::NodeOffsetBytes(std::uint64_t node_id) const {
    if (node_id >= metadata_.num_nodes) {
        throw std::runtime_error(BuildErrorMessage(
            "DiskannDiskIndexLayout::NodeOffsetBytes", "node_id is out of bounds."));
    }
    if (metadata_.nodes_per_sector > 0) {
        const std::uint64_t sector = node_id / metadata_.nodes_per_sector;
        const std::uint64_t offset_in_sector = node_id % metadata_.nodes_per_sector;
        return kSectorBytes + sector * kSectorBytes + offset_in_sector * metadata_.max_node_len;
    }
    return kSectorBytes + node_id * sectors_per_node() * kSectorBytes;
}

}  // namespace topoanns
