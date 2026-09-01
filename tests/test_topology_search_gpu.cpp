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
    const std::vector<std::uint32_t>& entries,
    const topoanns::PqDistanceOracle& oracle,
    const topoanns::TopologySearchParams& params) {
    std::vector<topoanns::RankedCandidate> candidates;
    std::unordered_set<std::uint32_t> visited;

    for (std::uint32_t node_id : entries) {
        if (node_id == topoanns::kInvalidNodeId || node_id >= resources.num_nodes()) {
            continue;
        }
        if (!visited.insert(node_id).second) {
            continue;
        }
        candidates.push_back(topoanns::RankedCandidate{oracle.Distance(0, node_id), node_id, false});
    }
    SortCandidates(candidates);

    topoanns::TopologySearchStats stats;
    stats.visited_nodes = visited.size();

    while (!candidates.empty() &&
           !topoanns::SearchStopCondition::TopLHasNoBetterUnexpanded(candidates, params.top_l) &&
           stats.expanded_nodes < params.max_expansions) {
        ++stats.iterations;
        std::vector<std::uint32_t> frontier;
        for (topoanns::RankedCandidate& candidate : candidates) {
            if (!candidate.valid() || candidate.expanded) {
                continue;
            }
            candidate.expanded = true;
            frontier.push_back(candidate.node_id);
            ++stats.expanded_nodes;
            if (frontier.size() == params.search_width ||
                stats.expanded_nodes == params.max_expansions) {
                break;
            }
        }
        if (frontier.empty()) {
            break;
        }

        for (std::uint32_t node_id : frontier) {
            for (std::uint32_t neighbor_id : resources.ReadHostNeighbors(node_id)) {
                if (neighbor_id == topoanns::kInvalidNodeId || neighbor_id >= resources.num_nodes()) {
                    continue;
                }
                if (!visited.insert(neighbor_id).second) {
                    continue;
                }
                candidates.push_back(
                    topoanns::RankedCandidate{oracle.Distance(0, neighbor_id), neighbor_id, false});
            }
        }
        stats.visited_nodes = visited.size();
        SortCandidates(candidates);
    }

    topoanns::TopologySearchResult result;
    for (const topoanns::RankedCandidate& candidate : candidates) {
        if (!candidate.valid()) {
            continue;
        }
        result.sorted_candidates.push_back(candidate);
        if (result.sorted_candidates.size() == params.top_l) {
            break;
        }
    }
    for (const topoanns::RankedCandidate& candidate : result.sorted_candidates) {
        result.topk.push_back(candidate);
        if (result.topk.size() == params.top_k) {
            break;
        }
    }
    result.stats = stats;
    return result;
}

}  // namespace

int main() {
    const auto pivots_path =
        std::filesystem::current_path() / "test_topology_search_gpu_pivots.bin";
    const auto codes_path =
        std::filesystem::current_path() / "test_topology_search_gpu_codes.bin";
    WriteSyntheticPqFiles(pivots_path, codes_path);

    const auto dataset = topoanns::TopologyLayout::MakePadded(
        6, {{1, 2}, {3}, {4}, {5}, {5}, {}});
    topoanns::SearchResources resources = topoanns::SearchResources::FromDataset(dataset);
    resources.LoadPqIndex(pivots_path, codes_path);

    topoanns::FixedEntryProvider entry_provider({0U});
    const std::vector<float> queries = {0.0f};
    const auto oracle = topoanns::PqDistanceOracle::FromFloatQueries(resources, queries, 1);

    topoanns::TopologySearchParams params;
    params.top_k = 2;
    params.top_l = 2;
    params.search_width = 1;
    params.max_expansions = 16;

    const auto gpu_result = topoanns::TopologySearch::Run(resources, entry_provider, oracle, 0,
                                                          params);
    const auto cpu_result = RunCpuReference(resources, {0U}, oracle, params);

    TOPOANNS_ASSERT_EQ(gpu_result.topk.size(), 2ULL);
    TOPOANNS_ASSERT_EQ(gpu_result.topk[0].node_id, 5U);
    TOPOANNS_ASSERT_EQ(gpu_result.topk[1].node_id, 3U);
    TOPOANNS_ASSERT_EQ(gpu_result.topk[0].distance, 1.0f);
    TOPOANNS_ASSERT_EQ(gpu_result.topk[1].distance, 4.0f);

    TOPOANNS_ASSERT_EQ(gpu_result.topk.size(), cpu_result.topk.size());
    TOPOANNS_ASSERT_EQ(gpu_result.sorted_candidates.size(), cpu_result.sorted_candidates.size());
    TOPOANNS_ASSERT_EQ(gpu_result.stats.visited_nodes, cpu_result.stats.visited_nodes);
    TOPOANNS_ASSERT_EQ(gpu_result.stats.expanded_nodes, cpu_result.stats.expanded_nodes);
    TOPOANNS_ASSERT_EQ(gpu_result.stats.iterations, cpu_result.stats.iterations);

    for (std::size_t i = 0; i < gpu_result.topk.size(); ++i) {
        TOPOANNS_ASSERT_EQ(gpu_result.topk[i].node_id, cpu_result.topk[i].node_id);
        TOPOANNS_ASSERT_EQ(gpu_result.topk[i].distance, cpu_result.topk[i].distance);
    }

    const std::size_t best_unexpanded_rank =
        topoanns::SearchStopCondition::BestUnexpandedRank(gpu_result.sorted_candidates);
    TOPOANNS_ASSERT_TRUE(best_unexpanded_rank >= params.top_l ||
                         best_unexpanded_rank == std::numeric_limits<std::size_t>::max());

    std::cout << "test_topology_search_gpu passed" << std::endl;
    return 0;
}
