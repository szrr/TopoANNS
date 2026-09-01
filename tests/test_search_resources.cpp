#include "topoanns/search_resources.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "test_support.hpp"
#include "topoanns/vector_page_layout.hpp"
#include "topoanns/vector_page_provider.hpp"
#include "topoanns/vector_store_builder.hpp"

namespace {

class FakeVectorPageProvider final : public topoanns::VectorPageProvider {
public:
    bool SupportsDeviceReads() const override { return true; }

    std::vector<std::uint8_t> ReadPages(const std::filesystem::path&,
                                        const std::vector<std::uint64_t>& page_ids,
                                        std::size_t,
                                        std::size_t page_size_bytes) const override {
        return std::vector<std::uint8_t>(page_ids.size() * page_size_bytes, 0);
    }

    topoanns::DevicePageReadResult ReadPagesToDevice(const std::filesystem::path&,
                                                     const std::vector<std::uint64_t>& page_ids,
                                                     std::size_t,
                                                     std::size_t page_size_bytes) const override {
        topoanns::DevicePageReadResult result;
        result.page_bytes = topoanns::CudaBuffer<std::uint8_t>::Allocate(
            page_ids.size() * page_size_bytes);
        return result;
    }
};

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

void WriteSyntheticErrorBounds(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::int32_t rows = 2;
    const std::int32_t cols = 1;
    const std::vector<float> bounds = {0.25f, 0.5f};
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(bounds.data()),
              static_cast<std::streamsize>(bounds.size() * sizeof(float)));
}

}  // namespace

int main() {
    using topoanns::SearchResources;
    using topoanns::TopologyLayout;

    const std::vector<std::vector<std::uint32_t>> adjacency_lists = {
        {1, 2},
        {0, 3},
        {3},
        {4},
        {},
    };

    const auto dataset = TopologyLayout::MakePadded(5, adjacency_lists);
    SearchResources resources = SearchResources::FromDataset(dataset);
    TOPOANNS_ASSERT_EQ(resources.num_nodes(), 5ULL);
    TOPOANNS_ASSERT_EQ(resources.degree(), 64U);
    TOPOANNS_ASSERT_TRUE(resources.host_topology_data() != nullptr);
    TOPOANNS_ASSERT_TRUE(resources.device_topology_data() != nullptr);
    TOPOANNS_ASSERT_TRUE(!resources.has_pq_index());
    TOPOANNS_ASSERT_TRUE(!resources.has_pq2_index());

    const auto neighbors = resources.ReadHostNeighbors(1);
    TOPOANNS_ASSERT_EQ(neighbors.size(), 64ULL);
    TOPOANNS_ASSERT_EQ(neighbors[0], 0U);
    TOPOANNS_ASSERT_EQ(neighbors[1], 3U);

    const std::filesystem::path out_path =
        std::filesystem::current_path() / "test_search_resources_topology.bin";
    TopologyLayout::WriteFile(out_path, dataset);
    const SearchResources loaded = SearchResources::FromTopologyFile(out_path);
    TOPOANNS_ASSERT_EQ(loaded.num_nodes(), resources.num_nodes());
    TOPOANNS_ASSERT_EQ(loaded.degree(), resources.degree());
    TOPOANNS_ASSERT_EQ(loaded.ReadHostNeighbors(2)[0], 3U);

    const auto pivots_path =
        std::filesystem::current_path() / "test_search_resources_pivots.bin";
    const auto codes_path =
        std::filesystem::current_path() / "test_search_resources_codes.bin";
    const auto error_path =
        std::filesystem::current_path() / "test_search_resources_error.bin";
    WriteSyntheticPqFiles(pivots_path, codes_path);
    WriteSyntheticErrorBounds(error_path);
    resources.LoadPqIndex(pivots_path, codes_path);
    TOPOANNS_ASSERT_TRUE(resources.has_pq_index());
    TOPOANNS_ASSERT_EQ(resources.pq_index().host().num_points, 2ULL);
    TOPOANNS_ASSERT_EQ(resources.pq_index().host().num_chunks, 1ULL);
    resources.LoadPq2Index(pivots_path, codes_path, error_path);
    TOPOANNS_ASSERT_TRUE(resources.has_pq2_index());
    TOPOANNS_ASSERT_EQ(resources.pq2_index().host().num_points, 2ULL);
    TOPOANNS_ASSERT_EQ(resources.pq2_index().host().num_chunks, 1ULL);
    TOPOANNS_ASSERT_TRUE(resources.pq2_is_residual_refine());
    TOPOANNS_ASSERT_TRUE(resources.has_pq2_error_bounds());
    TOPOANNS_ASSERT_EQ(resources.pq2_cross_terms().size(), 256ULL * 256ULL);
    const std::vector<float> error_bounds = resources.pq2_error_bounds_fp32().CopyToHost();
    TOPOANNS_ASSERT_EQ(error_bounds.size(), 2ULL);
    TOPOANNS_ASSERT_EQ(error_bounds[0], 0.25f);
    TOPOANNS_ASSERT_EQ(error_bounds[1], 0.5f);

    const auto vector_store_path =
        std::filesystem::current_path() / "test_search_resources_vectors.ssd";
    const auto layout =
        topoanns::VectorPageLayout::Create(4, topoanns::ScalarKind::kFloat32);
    const std::vector<std::vector<std::uint8_t>> vectors(
        2, std::vector<std::uint8_t>(layout.vector_bytes(), 0));
    topoanns::VectorStoreBuilder::WriteFile(vector_store_path, layout,
                                            topoanns::ScalarKind::kFloat32, 4, vectors);
    resources.LoadVectorStore(vector_store_path);
    TOPOANNS_ASSERT_TRUE(resources.has_vector_store());
    TOPOANNS_ASSERT_EQ(resources.vector_store_header().dim, 4U);
    TOPOANNS_ASSERT_EQ(resources.vector_store_layout().vector_bytes(), 16ULL);
    TOPOANNS_ASSERT_TRUE(resources.vector_store_path() == vector_store_path);
    bool threw = false;
    try {
        (void)resources.vector_page_provider();
    } catch (const std::runtime_error&) {
        threw = true;
    }
    TOPOANNS_ASSERT_TRUE(threw);

    resources.AttachVectorPageProvider(std::make_shared<FakeVectorPageProvider>());
    TOPOANNS_ASSERT_TRUE(resources.vector_page_provider().SupportsDeviceReads());

    std::cout << "test_search_resources passed" << std::endl;
    return 0;
}
