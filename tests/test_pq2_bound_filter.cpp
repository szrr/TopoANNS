#include "../src/search/fused_rerank_device.hpp"
#include "../src/search/topology_search_kernel.hpp"

#include "topoanns/search_resources.hpp"
#include "topoanns/vector_page_layout.hpp"
#include "topoanns/vector_store_builder.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "test_support.hpp"
#include "vector_page_test_provider.hpp"

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

void WritePqFiles(const std::filesystem::path& pivots_path,
                  const std::filesystem::path& codes_path,
                  const std::vector<float>& tables,
                  const std::vector<std::uint8_t>& codes) {
    constexpr std::uint64_t table_offset = 4096;
    constexpr std::uint64_t centroid_offset = table_offset + 8 + 256 * 2 * sizeof(float);
    constexpr std::uint64_t chunk_offset_offset = centroid_offset + 8 + 2 * sizeof(float);

    const std::vector<std::size_t> offsets = {
        static_cast<std::size_t>(table_offset),
        static_cast<std::size_t>(centroid_offset),
        static_cast<std::size_t>(chunk_offset_offset),
        0U,
    };
    const std::vector<float> centroid = {0.0f, 0.0f};
    const std::vector<std::uint32_t> chunk_offsets = {0U, 2U};

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
    const std::int32_t rows = 2;
    const std::int32_t cols = 1;
    compressed.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    compressed.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    compressed.write(reinterpret_cast<const char*>(codes.data()),
                     static_cast<std::streamsize>(codes.size()));
}

void WriteErrorBounds(const std::filesystem::path& path) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    const std::int32_t rows = 2;
    const std::int32_t cols = 1;
    const std::vector<float> values = {0.0f, 0.0f};
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(float)));
}

std::vector<std::uint8_t> FloatVectorBytes(float x, float y) {
    std::vector<float> values = {x, y};
    std::vector<std::uint8_t> bytes(values.size() * sizeof(float), 0);
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

}  // namespace

int main() {
    const auto base_pivots =
        std::filesystem::current_path() / "test_pq2_bound_base_pivots.bin";
    const auto base_codes =
        std::filesystem::current_path() / "test_pq2_bound_base_codes.bin";
    const auto residual_pivots =
        std::filesystem::current_path() / "test_pq2_bound_pivots.bin";
    const auto residual_codes =
        std::filesystem::current_path() / "test_pq2_bound_codes.bin";
    const auto error_path =
        std::filesystem::current_path() / "test_pq2_bound_error.bin";
    const auto vector_store_path =
        std::filesystem::current_path() / "test_pq2_bound_vectors.ssd";

    std::vector<float> base_tables(256 * 2, 0.0f);
    base_tables[1 * 2 + 0] = 1.0f;
    base_tables[1 * 2 + 1] = 0.0f;
    base_tables[2 * 2 + 0] = 0.0f;
    base_tables[2 * 2 + 1] = 2.0f;
    WritePqFiles(base_pivots, base_codes, base_tables, {1U, 2U});

    std::vector<float> residual_tables(256 * 2, 0.0f);
    residual_tables[1 * 2 + 0] = 0.0f;
    residual_tables[1 * 2 + 1] = 1.0f;
    residual_tables[2 * 2 + 0] = 1.0f;
    residual_tables[2 * 2 + 1] = 1.0f;
    WritePqFiles(residual_pivots, residual_codes, residual_tables, {1U, 2U});
    WriteErrorBounds(error_path);

    const auto dataset = topoanns::TopologyLayout::MakePadded(2, {{1}, {}});
    topoanns::SearchResources resources = topoanns::SearchResources::FromDataset(dataset);
    resources.LoadPqIndex(base_pivots, base_codes);
    resources.LoadPq2Index(residual_pivots, residual_codes, error_path);

    const topoanns::VectorPageLayout layout =
        topoanns::VectorPageLayout::Create(2, topoanns::ScalarKind::kFloat32);
    std::vector<std::vector<std::uint8_t>> vectors = {
        FloatVectorBytes(1.0f, 1.0f),
        FloatVectorBytes(1.0f, 3.0f),
    };
    topoanns::VectorStoreBuilder::WriteFile(vector_store_path, layout,
                                            topoanns::ScalarKind::kFloat32, 2, vectors);
    resources.LoadVectorStore(vector_store_path);
    resources.AttachVectorPageProvider(
        std::make_shared<FileBackedTestVectorPageProvider>(vector_store_path));

    topoanns::detail::DeviceTopologyBatchResult topology;
    topology.num_queries = 1;
    topology.candidate_capacity = 2;
    std::vector<topoanns::detail::DeviceTopologyCandidate> candidates(2);
    candidates[0].node_id = 1U;
    candidates[0].distance = 2.0f;
    candidates[1].node_id = 0U;
    candidates[1].distance = 1.0f;
    topology.candidate_buffer =
        topoanns::CudaBuffer<topoanns::detail::DeviceTopologyCandidate>::CopyFromHost(candidates);

    std::vector<topoanns::detail::DeviceTopologySearchStats> stats(1);
    stats[0].valid_candidates = 2;
    topology.stats_buffer =
        topoanns::CudaBuffer<topoanns::detail::DeviceTopologySearchStats>::CopyFromHost(stats);

    topoanns::RerankExactParams params;
    params.top_k = 1;
    params.top_n = 2;
    params.mode = topoanns::RerankExecutionMode::kPageByPage;
    params.rank_tile_size = 1;
    params.use_pq2_refine = true;
    params.use_pq2_bound_filter = true;
    params.pq2_refine_top_l = 2;

    const std::vector<float> queries = {1.0f, 1.0f};
    const auto rerank =
        topoanns::detail::RunBatchFloat32FromDeviceTopology(resources, topology, queries, 1, params);

    TOPOANNS_ASSERT_EQ(rerank.queries.size(), 1ULL);
    TOPOANNS_ASSERT_EQ(rerank.stats.exact_distance_count, 1ULL);
    TOPOANNS_ASSERT_EQ(rerank.stats.bound_filtered_count, 1ULL);
    TOPOANNS_ASSERT_EQ(rerank.queries[0].topk.size(), 1ULL);
    TOPOANNS_ASSERT_EQ(rerank.queries[0].topk[0].node_id, 0U);

    std::cout << "test_pq2_bound_filter passed" << std::endl;
    return 0;
}
