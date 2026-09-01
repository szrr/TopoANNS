#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/rerank_exact.hpp"
#include "topoanns/topology_search.hpp"
#include "topoanns/vector_page_layout.hpp"
#include "topoanns/vector_store_builder.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "test_support.hpp"
#include "vector_page_test_provider.hpp"
#include "topoanns/entry_provider.hpp"
#include "topoanns/search_resources.hpp"
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
    constexpr std::uint64_t centroid_offset = table_offset + 8 + 256 * sizeof(float);
    constexpr std::uint64_t chunk_offset_offset = centroid_offset + 8 + sizeof(float);

    const std::vector<std::size_t> offsets = {
        static_cast<std::size_t>(table_offset),
        static_cast<std::size_t>(centroid_offset),
        static_cast<std::size_t>(chunk_offset_offset),
        0U,
    };

    std::vector<float> tables(256, 0.0f);
    for (std::size_t i = 0; i < tables.size(); ++i) {
        tables[i] = static_cast<float>(i);
    }

    const std::vector<float> centroid = {0.0f};
    const std::vector<std::uint32_t> chunk_offsets = {0U, 1U};
    const std::vector<std::uint8_t> codes = {9U, 6U, 7U, 2U, 3U, 1U};

    std::ofstream pivots(pivots_path, std::ios::binary | std::ios::trunc);
    const std::vector<std::uint8_t> zero_pad(4096, 0);
    pivots.write(reinterpret_cast<const char*>(zero_pad.data()),
                 static_cast<std::streamsize>(zero_pad.size()));
    WriteBinBlock<std::size_t>(pivots, 0, 4, 1, offsets);
    WriteBinBlock<float>(pivots, table_offset, 256, 1, tables);
    WriteBinBlock<float>(pivots, centroid_offset, 1, 1, centroid);
    WriteBinBlock<std::uint32_t>(pivots, chunk_offset_offset, 2, 1, chunk_offsets);
    pivots.close();

    std::ofstream compressed(codes_path, std::ios::binary | std::ios::trunc);
    const std::int32_t rows = 6;
    const std::int32_t cols = 1;
    compressed.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    compressed.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    compressed.write(reinterpret_cast<const char*>(codes.data()),
                     static_cast<std::streamsize>(codes.size()));
    compressed.close();
}

std::vector<std::uint8_t> FloatVectorBytes(float value, std::size_t dim) {
    std::vector<float> values(dim, value);
    std::vector<std::uint8_t> bytes(values.size() * sizeof(float), 0);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

}  // namespace

int main() {
    const auto pivots_path =
        std::filesystem::current_path() / "test_rerank_exact_pivots.bin";
    const auto codes_path =
        std::filesystem::current_path() / "test_rerank_exact_codes.bin";
    const auto vector_store_path =
        std::filesystem::current_path() / "test_rerank_exact_vectors.ssd";
    WriteSyntheticPqFiles(pivots_path, codes_path);

    const auto dataset = topoanns::TopologyLayout::MakePadded(
        6, {{1, 2}, {3}, {4}, {5}, {5}, {}});
    topoanns::SearchResources resources = topoanns::SearchResources::FromDataset(dataset);
    resources.LoadPqIndex(pivots_path, codes_path);

    const topoanns::VectorPageLayout layout =
        topoanns::VectorPageLayout::Create(300, topoanns::ScalarKind::kFloat32);
    std::vector<std::vector<std::uint8_t>> vectors;
    for (float value : {9.0f, 6.0f, 7.0f, 2.0f, 3.0f, 1.0f}) {
        vectors.push_back(FloatVectorBytes(value, 300));
    }
    topoanns::VectorStoreBuilder::WriteFile(vector_store_path, layout,
                                            topoanns::ScalarKind::kFloat32, 300, vectors);
    resources.LoadVectorStore(vector_store_path);
    resources.AttachVectorPageProvider(
        std::make_shared<FileBackedTestVectorPageProvider>(vector_store_path));

    topoanns::FixedEntryProvider entry_provider({0U});
    const std::vector<float> queries(2 * 300, 0.0f);
    const auto oracle = topoanns::PqDistanceOracle::FromFloatQueries(resources, {0.0f, 0.0f}, 2);

    topoanns::TopologySearchParams search_params;
    search_params.top_k = 5;
    search_params.top_l = 5;
    search_params.search_width = 1;
    search_params.max_expansions = 16;
    const auto topology_results =
        topoanns::TopologySearch::RunBatch(resources, entry_provider, oracle, search_params);

    topoanns::RerankExactParams rerank_params;
    rerank_params.top_k = 2;
    rerank_params.top_n = 5;
    const auto rerank_result = topoanns::RerankExact::RunBatchFloat32(
        resources, topology_results, queries, 2, rerank_params);

    TOPOANNS_ASSERT_EQ(rerank_result.queries.size(), 2ULL);
    TOPOANNS_ASSERT_EQ(rerank_result.stats.exact_distance_count, 10ULL);
    TOPOANNS_ASSERT_EQ(rerank_result.stats.io_pages, 2ULL);

    TOPOANNS_ASSERT_EQ(rerank_result.queries[0].topk[0].node_id, 5U);
    TOPOANNS_ASSERT_EQ(rerank_result.queries[0].topk[1].node_id, 3U);
    TOPOANNS_ASSERT_EQ(rerank_result.queries[0].topk[0].distance, 300.0f);
    TOPOANNS_ASSERT_EQ(rerank_result.queries[0].topk[1].distance, 1200.0f);

    TOPOANNS_ASSERT_EQ(rerank_result.queries[1].topk[0].node_id, 5U);
    TOPOANNS_ASSERT_EQ(rerank_result.queries[1].topk[1].node_id, 3U);
    TOPOANNS_ASSERT_EQ(rerank_result.queries[1].topk[0].distance, 300.0f);
    TOPOANNS_ASSERT_EQ(rerank_result.queries[1].topk[1].distance, 1200.0f);

    std::cout << "test_rerank_exact passed" << std::endl;
    return 0;
}
