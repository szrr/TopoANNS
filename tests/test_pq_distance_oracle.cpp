#include "topoanns/pq_distance_oracle.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "test_support.hpp"
#include "topoanns/topology_layout.hpp"

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

    const std::vector<std::size_t> offsets = {
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
    tables[3 * 2 + 0] = 1.0f;
    tables[3 * 2 + 1] = 2.0f;

    const std::vector<float> centroid = {0.0f, 0.0f};
    const std::vector<std::uint32_t> chunk_offsets = {0U, 2U};
    const std::vector<std::uint8_t> codes = {1U, 2U, 3U};

    std::ofstream pivots(pivots_path, std::ios::binary | std::ios::trunc);
    const std::vector<std::uint8_t> zero_pad(4096, 0);
    pivots.write(reinterpret_cast<const char*>(zero_pad.data()),
                 static_cast<std::streamsize>(zero_pad.size()));
    WriteBinBlock<std::size_t>(pivots, 0, 4, 1, offsets);
    WriteBinBlock<float>(pivots, table_offset, 256, 2, tables);
    WriteBinBlock<float>(pivots, centroid_offset, 2, 1, centroid);
    WriteBinBlock<std::uint32_t>(pivots, chunk_offset_offset, 2, 1, chunk_offsets);
    pivots.close();

    std::ofstream compressed(codes_path, std::ios::binary | std::ios::trunc);
    const std::int32_t rows = 3;
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
        std::filesystem::current_path() / "test_pq_distance_oracle_pivots.bin";
    const auto codes_path =
        std::filesystem::current_path() / "test_pq_distance_oracle_codes.bin";
    WriteSyntheticPqFiles(pivots_path, codes_path);

    const auto dataset = topoanns::TopologyLayout::MakePadded(3, {{1}, {2}, {}});
    topoanns::SearchResources resources = topoanns::SearchResources::FromDataset(dataset);
    resources.LoadPqIndex(pivots_path, codes_path);

    const std::vector<float> queries = {
        1.0f, 2.0f,
        0.0f, 0.0f,
    };
    const auto oracle = topoanns::PqDistanceOracle::FromFloatQueries(resources, queries, 2);
    const auto direct_oracle =
        topoanns::PqDistanceOracle::FromFloatQueries(resources.pq_index(), queries, 2);

    TOPOANNS_ASSERT_EQ(oracle.num_queries(), 2ULL);
    TOPOANNS_ASSERT_EQ(oracle.Distance(0, 0), 4.0f);
    TOPOANNS_ASSERT_EQ(oracle.Distance(0, 1), 1.0f);
    TOPOANNS_ASSERT_EQ(oracle.Distance(0, 2), 0.0f);
    TOPOANNS_ASSERT_EQ(oracle.Distance(1, 0), 1.0f);
    TOPOANNS_ASSERT_EQ(oracle.Distance(1, 2), 5.0f);
    TOPOANNS_ASSERT_EQ(direct_oracle.Distance(0, 1), 1.0f);
    TOPOANNS_ASSERT_EQ(direct_oracle.Distance(1, 2), 5.0f);

    std::cout << "test_pq_distance_oracle passed" << std::endl;
    return 0;
}
