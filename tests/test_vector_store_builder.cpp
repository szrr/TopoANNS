#include "topoanns/vector_page_layout.hpp"
#include "topoanns/vector_store_builder.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "test_support.hpp"

int main() {
    using topoanns::ScalarKind;
    using topoanns::VectorPageLayout;
    using topoanns::VectorStoreBuilder;

    const VectorPageLayout layout =
        VectorPageLayout::Create(300, ScalarKind::kFloat32);
    TOPOANNS_ASSERT_EQ(layout.vector_bytes(), 1200ULL);
    TOPOANNS_ASSERT_EQ(layout.vectors_per_page(), 3ULL);

    std::vector<std::vector<std::uint8_t>> vectors(5);
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        vectors[i].resize(layout.vector_bytes());
        for (std::size_t j = 0; j < vectors[i].size(); ++j) {
            vectors[i][j] = static_cast<std::uint8_t>((i * 17 + j) % 251);
        }
    }

    const std::filesystem::path out_path =
        std::filesystem::current_path() / "test_vectors.ssd";
    VectorStoreBuilder::WriteFile(out_path, layout, ScalarKind::kFloat32, 300, vectors);

    const auto header = VectorStoreBuilder::ReadHeader(out_path);
    TOPOANNS_ASSERT_EQ(header.num_vectors, 5ULL);
    TOPOANNS_ASSERT_EQ(header.page_size_bytes, 4096ULL);
    TOPOANNS_ASSERT_EQ(header.vector_bytes, 1200ULL);
    TOPOANNS_ASSERT_EQ(header.vectors_per_page, 3ULL);

    const auto vec2 = VectorStoreBuilder::ReadVectorById(out_path, layout, 2);
    const auto vec3 = VectorStoreBuilder::ReadVectorById(out_path, layout, 3);
    TOPOANNS_ASSERT_EQ(vec2.size(), layout.vector_bytes());
    TOPOANNS_ASSERT_EQ(vec3.size(), layout.vector_bytes());
    TOPOANNS_ASSERT_EQ(vec2[0], vectors[2][0]);
    TOPOANNS_ASSERT_EQ(vec2.back(), vectors[2].back());
    TOPOANNS_ASSERT_EQ(vec3[0], vectors[3][0]);
    TOPOANNS_ASSERT_EQ(vec3.back(), vectors[3].back());

    std::ifstream in(out_path, std::ios::binary);
    TOPOANNS_ASSERT_TRUE(in.is_open());
    std::vector<std::uint8_t> first_page(4096, 0);
    in.seekg(4096, std::ios::beg);
    in.read(reinterpret_cast<char*>(first_page.data()),
            static_cast<std::streamsize>(first_page.size()));
    TOPOANNS_ASSERT_TRUE(in.good());

    for (std::size_t i = layout.payload_bytes_per_page(); i < first_page.size(); ++i) {
        TOPOANNS_ASSERT_EQ(first_page[i], 0U);
    }

    const std::uint64_t expected_pages = layout.PageCountForVectors(vectors.size());
    const std::uint64_t expected_file_size = 4096ULL + expected_pages * 4096ULL;
    TOPOANNS_ASSERT_EQ(std::filesystem::file_size(out_path), expected_file_size);

    std::cout << "test_vector_store_builder passed" << std::endl;
    return 0;
}
