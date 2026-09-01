#include "topoanns/pq_index.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "test_support.hpp"

namespace {

template <typename T>
void WriteBinBlock(std::ofstream& out,
                   std::uint64_t offset,
                   std::int32_t rows,
                   std::int32_t cols,
                   const std::vector<T>& data) {
    out.seekp(static_cast<std::streamoff>(offset), std::ios::beg);
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()),
                  static_cast<std::streamsize>(data.size() * sizeof(T)));
    }
}

void WriteSyntheticPqFiles(const std::filesystem::path& pivots_path,
                           const std::filesystem::path& codes_path) {
    constexpr std::uint64_t table_offset = 4096;
    constexpr std::uint64_t centroid_offset = table_offset + 8 + 256 * 2 * sizeof(float);
    constexpr std::uint64_t chunk_offset_offset = centroid_offset + 8 + 2 * sizeof(float);

    std::vector<std::size_t> offsets = {
        static_cast<std::size_t>(table_offset),
        static_cast<std::size_t>(centroid_offset),
        static_cast<std::size_t>(chunk_offset_offset),
        0U,
    };

    std::vector<float> tables(256 * 2, 0.0f);
    tables[0 * 2 + 0] = 0.0f;
    tables[0 * 2 + 1] = 0.0f;
    tables[1 * 2 + 0] = 1.0f;
    tables[1 * 2 + 1] = 0.0f;
    tables[2 * 2 + 0] = 0.0f;
    tables[2 * 2 + 1] = 2.0f;

    std::vector<float> centroid = {0.0f, 0.0f};
    std::vector<std::uint32_t> chunk_offsets = {0U, 2U};
    std::vector<std::uint8_t> codes = {1U, 2U};

    std::ofstream pivots(pivots_path, std::ios::binary | std::ios::trunc);
    std::vector<std::uint8_t> zero_pad(4096, 0);
    pivots.write(reinterpret_cast<const char*>(zero_pad.data()),
                 static_cast<std::streamsize>(zero_pad.size()));
    WriteBinBlock<std::size_t>(pivots, 0, 4, 1, offsets);
    WriteBinBlock<float>(pivots, table_offset, 256, 2, tables);
    WriteBinBlock<float>(pivots, centroid_offset, 2, 1, centroid);
    WriteBinBlock<std::uint32_t>(pivots, chunk_offset_offset, 2, 1, chunk_offsets);
    pivots.close();

    std::ofstream compressed(codes_path, std::ios::binary | std::ios::trunc);
    const std::int32_t rows = 2;
    const std::int32_t cols = 1;
    compressed.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    compressed.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    compressed.write(reinterpret_cast<const char*>(codes.data()),
                     static_cast<std::streamsize>(codes.size()));
    compressed.close();
}

}  // namespace

int main() {
    const auto pivots_path =
        std::filesystem::current_path() / "test_pq_loader_pivots.bin";
    const auto codes_path =
        std::filesystem::current_path() / "test_pq_loader_codes.bin";
    WriteSyntheticPqFiles(pivots_path, codes_path);

    const topoanns::PqIndex pq_index =
        topoanns::PqIndex::LoadFromSeparatePaths(pivots_path, codes_path);
    const topoanns::PqIndex mapped_pq_index =
        topoanns::PqIndex::LoadFromSeparatePathsMapped(pivots_path, codes_path);

    TOPOANNS_ASSERT_EQ(pq_index.host().num_points, 2ULL);
    TOPOANNS_ASSERT_EQ(pq_index.host().ndims, 2ULL);
    TOPOANNS_ASSERT_EQ(pq_index.host().num_chunks, 1ULL);
    TOPOANNS_ASSERT_EQ(pq_index.host().codes.size(), 2ULL);
    TOPOANNS_ASSERT_EQ(pq_index.host().codes[0], 1U);
    TOPOANNS_ASSERT_EQ(pq_index.host().codes[1], 2U);
    TOPOANNS_ASSERT_EQ(pq_index.host().chunk_offsets.size(), 2ULL);
    TOPOANNS_ASSERT_EQ(pq_index.host().chunk_offsets[1], 2U);
    TOPOANNS_ASSERT_EQ(pq_index.host().tables_row_major[2], 1.0f);
    TOPOANNS_ASSERT_EQ(pq_index.host().tables_row_major[5], 2.0f);
    TOPOANNS_ASSERT_EQ(pq_index.host().tables_col_major[1], 1.0f);
    TOPOANNS_ASSERT_EQ(pq_index.host().tables_col_major[2 + 256], 2.0f);
    TOPOANNS_ASSERT_TRUE(pq_index.device().codes.get() != nullptr);
    TOPOANNS_ASSERT_TRUE(pq_index.device().tables_col_major.get() != nullptr);
    TOPOANNS_ASSERT_TRUE(mapped_pq_index.codes_are_mapped());
    TOPOANNS_ASSERT_TRUE(mapped_pq_index.host().codes_on_host);
    TOPOANNS_ASSERT_TRUE(mapped_pq_index.host_codes() != nullptr);
    TOPOANNS_ASSERT_TRUE(mapped_pq_index.device_codes() != nullptr);
    TOPOANNS_ASSERT_EQ(mapped_pq_index.host_codes()[0], 1U);
    TOPOANNS_ASSERT_EQ(mapped_pq_index.host_codes()[1], 2U);

    std::cout << "test_pq_loader passed" << std::endl;
    return 0;
}
