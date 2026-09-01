#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "topoanns/common.hpp"

namespace topoanns {

struct DiskannDiskIndexMetadata {
    std::uint64_t num_nodes = 0;
    std::uint64_t vector_dim = 0;
    std::uint64_t medoid_id = 0;
    std::uint64_t max_node_len = 0;
    std::uint64_t nodes_per_sector = 0;
    std::uint64_t num_frozen_points = 0;
    std::uint64_t file_num_frozen_points = 0;
    std::uint64_t append_reorder = 0;
    std::uint64_t reorder_data_start = 0;
};

class DiskannDiskIndexLayout {
public:
    static constexpr std::size_t kSectorBytes = kDefaultPageSizeBytes;

    explicit DiskannDiskIndexLayout(DiskannDiskIndexMetadata metadata)
        : metadata_(metadata) {}

    static DiskannDiskIndexLayout Load(const std::filesystem::path& path);

    const DiskannDiskIndexMetadata& metadata() const { return metadata_; }
    std::size_t coord_bytes() const;
    std::size_t neighbor_capacity() const;
    std::uint64_t sectors_per_node() const;
    std::uint64_t NodeOffsetBytes(std::uint64_t node_id) const;

private:
    DiskannDiskIndexMetadata metadata_{};
};

}  // namespace topoanns
