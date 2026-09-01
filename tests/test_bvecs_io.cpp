#include "topoanns/bvecs_io.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "test_support.hpp"

namespace {

void WriteRecord(std::ofstream& out, const std::vector<std::uint8_t>& values) {
    const std::uint32_t dim = static_cast<std::uint32_t>(values.size());
    out.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size()));
}

}  // namespace

int main() {
    const auto path = std::filesystem::current_path() / "test_bvecs_io.bvecs";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    WriteRecord(out, {1, 2, 3});
    WriteRecord(out, {4, 5, 6});
    WriteRecord(out, {7, 8, 9});
    out.close();

    const auto metadata = topoanns::ReadBvecsMetadata(path);
    TOPOANNS_ASSERT_EQ(metadata.dim, 3U);
    TOPOANNS_ASSERT_EQ(metadata.record_bytes, 7U);
    TOPOANNS_ASSERT_EQ(metadata.num_vectors, 3ULL);

    const auto vectors = topoanns::ReadBvecsRangeAsFloat32(path, 1, 2);
    TOPOANNS_ASSERT_EQ(vectors.size(), 6ULL);
    TOPOANNS_ASSERT_EQ(vectors[0], 4.0f);
    TOPOANNS_ASSERT_EQ(vectors[1], 5.0f);
    TOPOANNS_ASSERT_EQ(vectors[2], 6.0f);
    TOPOANNS_ASSERT_EQ(vectors[3], 7.0f);
    TOPOANNS_ASSERT_EQ(vectors[4], 8.0f);
    TOPOANNS_ASSERT_EQ(vectors[5], 9.0f);

    std::cout << "test_bvecs_io passed" << std::endl;
    return 0;
}
