#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/topology_search.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>

#include "test_support.hpp"
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

void SortCandidates(std::vector<topoanns::RankedCandidate>& candidates) {
    std::sort(candidates.begin(), candidates.end(),
              [](const topoanns::RankedCandidate& lhs,
                 const topoanns::RankedCandidate& rhs) {
                  if (lhs.valid() != rhs.valid()) {
                      return lhs.valid() > rhs.valid();
                  }
                  if (!lhs.valid() && !rhs.valid()) {
                      return false;
                  }
                  if (lhs.distance != rhs.distance) {
                      return lhs.distance < rhs.distance;
                  }
                  return lhs.node_id < rhs.node_id;
              });
}

topoanns::TopologySearchResult RunCpuReference(
    const topoanns::SearchResources& resources,
    const topoanns::PqDistanceOracle& oracle,
    std::size_t query_id,
    const topoanns::TopologySearchParams& params) {
    std::vector<topoanns::RankedCandidate> candidates;
    std::unordered_set<std::uint32_t> visited;
    visited.insert(0U);
    candidates.push_back({oracle.Distance(query_id, 0U), 0U, false});
    SortCandidates(candidates);

    topoanns::TopologySearchStats stats;
    stats.visited_nodes = 1;
    while (!candidates.empty() &&
           !topoanns::SearchStopCondition::TopLHasNoBetterUnexpanded(candidates, params.top_l) &&
           stats.expanded_nodes < params.max_expansions) {
        ++stats.iterations;
        std::vector<std::uint32_t> frontier;
        for (auto& candidate : candidates) {
            if (!candidate.valid() || candidate.expanded) {
                continue;
            }
            candidate.expanded = true;
            frontier.push_back(candidate.node_id);
            ++stats.expanded_nodes;
            if (frontier.size() == params.search_width) {
                break;
            }
        }
        for (std::uint32_t node_id : frontier) {
            for (std::uint32_t neighbor_id : resources.ReadHostNeighbors(node_id)) {
                if (neighbor_id == topoanns::kInvalidNodeId || neighbor_id >= resources.num_nodes()) {
                    continue;
                }
                if (!visited.insert(neighbor_id).second) {
                    continue;
                }
                candidates.push_back({oracle.Distance(query_id, neighbor_id), neighbor_id, false});
            }
        }
        stats.visited_nodes = visited.size();
        SortCandidates(candidates);
    }

    topoanns::TopologySearchResult result;
    result.stats = stats;
    for (const auto& candidate : candidates) {
        if (!candidate.valid()) {
            continue;
        }
        result.sorted_candidates.push_back(candidate);
        if (result.sorted_candidates.size() == params.top_l) {
            break;
        }
    }
    for (const auto& candidate : result.sorted_candidates) {
        result.topk.push_back(candidate);
        if (result.topk.size() == params.top_k) {
            break;
        }
    }
    return result;
}

}  // namespace

int main() {
    const auto pivots_path =
        std::filesystem::current_path() / "test_topology_search_batch_gpu_pivots.bin";
    const auto codes_path =
        std::filesystem::current_path() / "test_topology_search_batch_gpu_codes.bin";
    WriteSyntheticPqFiles(pivots_path, codes_path);

    const auto dataset = topoanns::TopologyLayout::MakePadded(
        6, {{1, 2}, {3}, {4}, {5}, {5}, {}});
    topoanns::SearchResources resources = topoanns::SearchResources::FromDataset(dataset);
    resources.LoadPqIndex(pivots_path, codes_path);

    topoanns::FixedEntryProvider entry_provider({0U});
    const std::vector<float> queries = {0.0f, 8.0f};
    const auto oracle = topoanns::PqDistanceOracle::FromFloatQueries(resources, queries, 2);

    topoanns::TopologySearchParams params;
    params.top_k = 2;
    params.top_l = 2;
    params.search_width = 1;
    params.max_expansions = 16;

    const auto batch_results =
        topoanns::TopologySearch::RunBatch(resources, entry_provider, oracle, params);
    const auto single_q1 =
        topoanns::TopologySearch::Run(resources, entry_provider, oracle, 1, params);
    TOPOANNS_ASSERT_EQ(batch_results.size(), 2ULL);

    const auto cpu0 = RunCpuReference(resources, oracle, 0, params);
    const auto cpu1 = RunCpuReference(resources, oracle, 1, params);
    TOPOANNS_ASSERT_EQ(batch_results[0].topk[0].node_id, 5U);
    TOPOANNS_ASSERT_EQ(batch_results[0].topk[1].node_id, 3U);
    TOPOANNS_ASSERT_EQ(batch_results[1].topk[0].node_id, 0U);
    TOPOANNS_ASSERT_EQ(batch_results[1].topk[1].node_id, 2U);
    TOPOANNS_ASSERT_EQ(single_q1.topk[0].node_id, 0U);
    TOPOANNS_ASSERT_EQ(single_q1.topk[1].node_id, 2U);
    TOPOANNS_ASSERT_EQ(batch_results[0].stats.iterations, cpu0.stats.iterations);
    TOPOANNS_ASSERT_EQ(batch_results[1].stats.iterations, cpu1.stats.iterations);
    TOPOANNS_ASSERT_EQ(batch_results[0].stats.visited_nodes, cpu0.stats.visited_nodes);
    TOPOANNS_ASSERT_EQ(batch_results[1].stats.expanded_nodes, cpu1.stats.expanded_nodes);

    std::cout << "test_topology_search_batch_gpu passed" << std::endl;
    return 0;
}
