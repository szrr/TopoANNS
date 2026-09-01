#include "../src/search/fused_rerank_device.hpp"
#include "../src/search/topology_search_kernel.hpp"

#include "topoanns/search_resources.hpp"
#include "topoanns/topology_layout.hpp"

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

}  // namespace

int main() {
    const auto base_pivots =
        std::filesystem::current_path() / "test_pq2_residual_base_pivots.bin";
    const auto base_codes =
        std::filesystem::current_path() / "test_pq2_residual_base_codes.bin";
    const auto residual_pivots =
        std::filesystem::current_path() / "test_pq2_residual_pivots.bin";
    const auto residual_codes =
        std::filesystem::current_path() / "test_pq2_residual_codes.bin";
    const auto error_path =
        std::filesystem::current_path() / "test_pq2_residual_error.bin";

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

    const std::vector<float> queries = {1.0f, 1.0f};
    const auto refined = topoanns::detail::RunPq2RefineBatchDevice(resources, topology, queries, 1, 2);
    const std::vector<topoanns::detail::DeviceTopologyCandidate> host_refined =
        refined.candidate_buffer.CopyToHost();

    TOPOANNS_ASSERT_EQ(host_refined.size(), 2ULL);
    TOPOANNS_ASSERT_EQ(host_refined[0].node_id, 0U);
    TOPOANNS_ASSERT_EQ(host_refined[1].node_id, 1U);
    TOPOANNS_ASSERT_EQ(host_refined[0].distance, 2.0f);
    TOPOANNS_ASSERT_EQ(host_refined[1].distance, 6.0f);

    std::cout << "test_pq2_residual_refine passed" << std::endl;
    return 0;
}
