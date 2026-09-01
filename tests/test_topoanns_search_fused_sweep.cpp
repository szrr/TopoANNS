#include "topoanns/device_entry_batch.hpp"
#include "topoanns/topoanns_search.hpp"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "test_support.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/topology_layout.hpp"
#include "topoanns/vector_page_layout.hpp"
#include "topoanns/vector_store_builder.hpp"
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

void WriteSyntheticPqFiles(const std::filesystem::path& pivots_path,
                           const std::filesystem::path& codes_path,
                           std::size_t dim) {
    constexpr std::uint64_t table_offset = 4096;
    const std::uint64_t centroid_offset =
        table_offset + 8 + static_cast<std::uint64_t>(256 * dim * sizeof(float));
    const std::uint64_t chunk_offset_offset =
        centroid_offset + 8 + static_cast<std::uint64_t>(dim * sizeof(float));

    const std::vector<std::size_t> offsets = {
        static_cast<std::size_t>(table_offset),
        static_cast<std::size_t>(centroid_offset),
        static_cast<std::size_t>(chunk_offset_offset),
        0U,
    };

    std::vector<float> tables(256 * dim, 0.0f);
    for (std::size_t center = 0; center < 256; ++center) {
        for (std::size_t dim_idx = 0; dim_idx < dim; ++dim_idx) {
            tables[center * dim + dim_idx] = static_cast<float>(center);
        }
    }

    const std::vector<float> centroid(dim, 0.0f);
    const std::vector<std::uint32_t> chunk_offsets = {0U, static_cast<std::uint32_t>(dim)};
    const std::vector<std::uint8_t> codes = {9U, 6U, 7U, 2U, 3U, 1U};

    std::ofstream pivots(pivots_path, std::ios::binary | std::ios::trunc);
    const std::vector<std::uint8_t> zero_pad(4096, 0);
    pivots.write(reinterpret_cast<const char*>(zero_pad.data()),
                 static_cast<std::streamsize>(zero_pad.size()));
    WriteBinBlock<std::size_t>(pivots, 0, 4, 1, offsets);
    WriteBinBlock<float>(pivots, table_offset, 256, static_cast<std::int32_t>(dim), tables);
    WriteBinBlock<float>(pivots, centroid_offset, static_cast<std::int32_t>(dim), 1, centroid);
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

std::vector<float> ConstantQueries(const std::vector<float>& values, std::size_t dim) {
    std::vector<float> queries;
    queries.reserve(values.size() * dim);
    for (const float value : values) {
        queries.insert(queries.end(), dim, value);
    }
    return queries;
}

topoanns::DeviceEntryBatch BuildFixedDeviceEntries(std::size_t num_queries,
                                                   std::uint32_t entry_id) {
    topoanns::DeviceEntryBatch batch;
    batch.num_queries = num_queries;
    batch.entries_per_query = 1;
    std::vector<std::uint32_t> offsets(num_queries + 1, 0U);
    std::vector<std::uint32_t> ids(num_queries, entry_id);
    for (std::size_t i = 0; i <= num_queries; ++i) {
        offsets[i] = static_cast<std::uint32_t>(i);
    }
    batch.offsets = topoanns::CudaBuffer<std::uint32_t>::CopyFromHost(offsets);
    batch.ids = topoanns::CudaBuffer<std::uint32_t>::CopyFromHost(ids);
    return batch;
}

}  // namespace

int main() {
    constexpr std::size_t kDim = 300;
    const auto pivots_path =
        std::filesystem::current_path() / "test_topoanns_search_fused_sweep_pivots.bin";
    const auto codes_path =
        std::filesystem::current_path() / "test_topoanns_search_fused_sweep_codes.bin";
    const auto vector_store_path =
        std::filesystem::current_path() / "test_topoanns_search_fused_sweep_vectors.ssd";
    WriteSyntheticPqFiles(pivots_path, codes_path, kDim);

    const auto dataset = topoanns::TopologyLayout::MakePadded(
        6, {{1, 2}, {3}, {4}, {5}, {5}, {}});
    topoanns::SearchResources resources = topoanns::SearchResources::FromDataset(dataset);
    resources.LoadPqIndex(pivots_path, codes_path);

    const topoanns::VectorPageLayout layout =
        topoanns::VectorPageLayout::Create(kDim, topoanns::ScalarKind::kFloat32);
    std::vector<std::vector<std::uint8_t>> vectors;
    for (const float value : {9.0f, 6.0f, 7.0f, 2.0f, 3.0f, 1.0f}) {
        vectors.push_back(FloatVectorBytes(value, kDim));
    }
    topoanns::VectorStoreBuilder::WriteFile(vector_store_path, layout,
                                            topoanns::ScalarKind::kFloat32,
                                            static_cast<std::uint32_t>(kDim), vectors);
    resources.LoadVectorStore(vector_store_path);
    resources.AttachVectorPageProvider(
        std::make_shared<FileBackedTestVectorPageProvider>(vector_store_path));

    topoanns::DeviceEntryBatch device_entries = BuildFixedDeviceEntries(2, 0U);
    const std::vector<float> queries = ConstantQueries({0.0f, 8.0f}, kDim);

    topoanns::TopologySearchParams topology_params;
    topology_params.top_k = 5;
    topology_params.top_l = 4;
    topology_params.search_width = 1;
    topology_params.max_expansions = 16;

    topoanns::RerankExactParams rerank_small;
    rerank_small.top_k = 2;
    rerank_small.top_n = 2;
    rerank_small.mode = topoanns::RerankExecutionMode::kTiled;

    topoanns::RerankExactParams rerank_large;
    rerank_large.top_k = 2;
    rerank_large.top_n = 4;
    rerank_large.mode = topoanns::RerankExecutionMode::kTiled;

    topoanns::TopoAnnsSearchParams baseline_params;
    baseline_params.topology = topology_params;
    baseline_params.rerank = rerank_large;

    const auto baseline = topoanns::TopoAnnsSearch::RunBatchFloat32Fused(
        resources, device_entries, queries, 2, baseline_params);
    const auto sweep = topoanns::TopoAnnsSearch::RunBatchFloat32FusedRerankSweep(
        resources, device_entries, queries, 2, topology_params, {rerank_small, rerank_large});

    TOPOANNS_ASSERT_EQ(sweep.size(), 2U);
    TOPOANNS_ASSERT_EQ(sweep[0].queries.size(), 2U);
    TOPOANNS_ASSERT_EQ(sweep[1].queries.size(), baseline.queries.size());

    for (std::size_t query_id = 0; query_id < baseline.queries.size(); ++query_id) {
        TOPOANNS_ASSERT_EQ(sweep[1].queries[query_id].topology.stats.visited_nodes,
                           baseline.queries[query_id].topology.stats.visited_nodes);
        TOPOANNS_ASSERT_EQ(sweep[1].queries[query_id].topology.stats.expanded_nodes,
                           baseline.queries[query_id].topology.stats.expanded_nodes);
        TOPOANNS_ASSERT_EQ(sweep[1].queries[query_id].rerank.topk.size(),
                           baseline.queries[query_id].rerank.topk.size());
        for (std::size_t rank = 0; rank < baseline.queries[query_id].rerank.topk.size(); ++rank) {
            TOPOANNS_ASSERT_EQ(sweep[1].queries[query_id].rerank.topk[rank].node_id,
                               baseline.queries[query_id].rerank.topk[rank].node_id);
            TOPOANNS_ASSERT_EQ(sweep[1].queries[query_id].rerank.topk[rank].distance,
                               baseline.queries[query_id].rerank.topk[rank].distance);
        }
        TOPOANNS_ASSERT_TRUE(sweep[0].queries[query_id].rerank.topk.size() <=
                             sweep[1].queries[query_id].rerank.topk.size());
    }

    std::cout << "test_topoanns_search_fused_sweep passed" << std::endl;
    return 0;
}
