#include "topoanns/diskann_disk_index.hpp"

#include <array>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "test_support.hpp"

namespace {

constexpr std::size_t kSectorBytes = topoanns::kDefaultPageSizeBytes;

void WriteNode(char* dst,
               const std::array<float, 2>& vec,
               std::uint32_t degree,
               const std::array<std::uint32_t, 4>& neighbors) {
    std::memcpy(dst, vec.data(), sizeof(float) * vec.size());
    std::memcpy(dst + sizeof(float) * vec.size(), &degree, sizeof(degree));
    std::memcpy(dst + sizeof(float) * vec.size() + sizeof(degree),
                neighbors.data(),
                sizeof(std::uint32_t) * neighbors.size());
}

}  // namespace

int main() {
    const auto path = std::filesystem::current_path() / "test_diskann_disk.index";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);

    const std::uint32_t meta_count = 9;
    const std::uint32_t meta_dim = 1;
    const std::array<std::uint64_t, 9> meta = {
        3ULL,
        2ULL,
        1ULL,
        28ULL,
        2ULL,
        0ULL,
        0ULL,
        0ULL,
        0ULL,
    };
    out.write(reinterpret_cast<const char*>(&meta_count), sizeof(meta_count));
    out.write(reinterpret_cast<const char*>(&meta_dim), sizeof(meta_dim));
    out.write(reinterpret_cast<const char*>(meta.data()),
              static_cast<std::streamsize>(sizeof(meta)));

    std::array<char, kSectorBytes - 8 - sizeof(meta)> zero_header_tail{};
    out.write(zero_header_tail.data(),
              static_cast<std::streamsize>(zero_header_tail.size()));

    std::array<char, kSectorBytes> sector{};
    WriteNode(sector.data() + 0 * 28, {1.0f, 2.0f}, 2, {1U, 2U, 99U, 100U});
    WriteNode(sector.data() + 1 * 28, {3.0f, 4.0f}, 1, {0U, 7U, 8U, 9U});
    out.write(sector.data(), static_cast<std::streamsize>(sector.size()));

    std::array<char, kSectorBytes> sector2{};
    WriteNode(sector2.data() + 0 * 28, {5.0f, 6.0f}, 3, {0U, 1U, 2U, 77U});
    out.write(sector2.data(), static_cast<std::streamsize>(sector2.size()));
    out.close();

    const auto layout = topoanns::DiskannDiskIndexLayout::Load(path);
    TOPOANNS_ASSERT_EQ(layout.metadata().num_nodes, 3ULL);
    TOPOANNS_ASSERT_EQ(layout.metadata().vector_dim, 2ULL);
    TOPOANNS_ASSERT_EQ(layout.metadata().medoid_id, 1ULL);
    TOPOANNS_ASSERT_EQ(layout.metadata().max_node_len, 28ULL);
    TOPOANNS_ASSERT_EQ(layout.metadata().nodes_per_sector, 2ULL);
    TOPOANNS_ASSERT_EQ(layout.coord_bytes(), 8ULL);
    TOPOANNS_ASSERT_EQ(layout.neighbor_capacity(), 4ULL);
    TOPOANNS_ASSERT_EQ(layout.sectors_per_node(), 1ULL);
    TOPOANNS_ASSERT_EQ(layout.NodeOffsetBytes(0), 4096ULL);
    TOPOANNS_ASSERT_EQ(layout.NodeOffsetBytes(1), 4096ULL + 28ULL);
    TOPOANNS_ASSERT_EQ(layout.NodeOffsetBytes(2), 8192ULL);

    std::cout << "test_diskann_disk_index passed" << std::endl;
    return 0;
}
