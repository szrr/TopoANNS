#include "../src/search/topology_search_kernel.hpp"

#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/topology_layout.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

using topoanns::detail::DeviceTopologyCandidate;
using topoanns::detail::DeviceTopologyDebugConfig;
using topoanns::detail::DeviceTopologyDebugSnapshot;
using topoanns::detail::DeviceTopologyDebugTrace;

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
                           std::size_t num_nodes,
                           std::uint32_t seed) {
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

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> code_dist(0, 255);
    std::vector<std::uint8_t> codes(num_nodes);
    for (std::size_t i = 0; i < num_nodes; ++i) {
        codes[i] = static_cast<std::uint8_t>(code_dist(rng));
    }

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
    const std::int32_t rows = static_cast<std::int32_t>(num_nodes);
    const std::int32_t cols = 1;
    compressed.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    compressed.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    compressed.write(reinterpret_cast<const char*>(codes.data()),
                     static_cast<std::streamsize>(codes.size()));
    compressed.close();
}

topoanns::TopologyDataset MakeRandomDataset(std::size_t num_nodes,
                                            std::size_t degree,
                                            std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<std::uint32_t> node_dist(
        0U, static_cast<std::uint32_t>(num_nodes - 1U));
    std::vector<std::vector<std::uint32_t>> adjacency(num_nodes);
    for (std::size_t node = 0; node < num_nodes; ++node) {
        std::unordered_set<std::uint32_t> neighbors;
        while (neighbors.size() < degree) {
            const std::uint32_t neighbor = node_dist(rng);
            if (neighbor == node) {
                continue;
            }
            neighbors.insert(neighbor);
        }
        adjacency[node].assign(neighbors.begin(), neighbors.end());
    }
    return topoanns::TopologyLayout::MakePadded(num_nodes, adjacency);
}

topoanns::DeviceEntryBatch MakeEntryBatch(const std::vector<std::uint32_t>& entries) {
    topoanns::DeviceEntryBatch batch;
    batch.num_queries = 1;
    batch.entries_per_query = entries.size();
    batch.offsets = topoanns::CudaBuffer<std::uint32_t>::CopyFromHost(
        std::vector<std::uint32_t>{0U, static_cast<std::uint32_t>(entries.size())});
    batch.ids = topoanns::CudaBuffer<std::uint32_t>::CopyFromHost(entries);
    return batch;
}

void PrintCandidate(const DeviceTopologyCandidate& candidate, std::size_t index,
                    const char* tag) {
    std::cout << tag << "[" << index << "] node=" << candidate.raw_node_id()
              << " dist=" << candidate.distance
              << " expanded=" << static_cast<std::uint32_t>(candidate.expanded()) << "\n";
}

bool SnapshotEqual(const DeviceTopologyDebugTrace& lhs,
                   const DeviceTopologyDebugTrace& rhs,
                   std::size_t snapshot_index,
                   std::size_t* first_candidate_diff) {
    const std::size_t prefix = lhs.capture_prefix;
    const std::size_t base = snapshot_index * prefix;
    for (std::size_t i = 0; i < prefix; ++i) {
        const auto& lc = lhs.candidate_snapshots[base + i];
        const auto& rc = rhs.candidate_snapshots[base + i];
        if (lc.raw_node_id() != rc.raw_node_id() || lc.distance != rc.distance ||
            lc.expanded() != rc.expanded()) {
            *first_candidate_diff = i;
            return false;
        }
    }
    *first_candidate_diff = prefix;
    return true;
}

bool MetadataEqual(const DeviceTopologyDebugSnapshot& lhs,
                   const DeviceTopologyDebugSnapshot& rhs) {
    return lhs.merge_ordinal == rhs.merge_ordinal &&
           lhs.phase == rhs.phase &&
           lhs.search_iteration == rhs.search_iteration &&
           lhs.frontier_valid == rhs.frontier_valid &&
           lhs.accepted_frontier == rhs.accepted_frontier &&
           lhs.selected_count == rhs.selected_count &&
           lhs.valid_candidates == rhs.valid_candidates &&
           lhs.visited_nodes == rhs.visited_nodes &&
           lhs.expanded_nodes == rhs.expanded_nodes &&
           lhs.frontier_checksum == rhs.frontier_checksum &&
           lhs.visited_hash_checksum == rhs.visited_hash_checksum;
}

void PrintSnapshotMetadata(const char* label, const DeviceTopologyDebugSnapshot& snapshot) {
    std::cout << label << " merge=" << snapshot.merge_ordinal
              << " phase=" << snapshot.phase
              << " iter=" << snapshot.search_iteration
              << " frontier_valid=" << snapshot.frontier_valid
              << " accepted=" << snapshot.accepted_frontier
              << " selected=" << snapshot.selected_count
              << " valid=" << snapshot.valid_candidates
              << " visited=" << snapshot.visited_nodes
              << " expanded=" << snapshot.expanded_nodes
              << " frontier_checksum=" << snapshot.frontier_checksum
              << " visited_hash_checksum=" << snapshot.visited_hash_checksum << "\n";
}

bool FindFirstDiff(const DeviceTopologyDebugTrace& low_trace,
                   const DeviceTopologyDebugTrace& high_trace,
                   std::size_t* snapshot_index,
                   std::size_t* candidate_index) {
    const std::size_t low_count = low_trace.snapshot_counts[0];
    const std::size_t high_count = high_trace.snapshot_counts[0];
    const std::size_t common = std::min(low_count, high_count);
    for (std::size_t i = 0; i < common; ++i) {
        if (!MetadataEqual(low_trace.snapshots[i], high_trace.snapshots[i])) {
            *snapshot_index = i;
            *candidate_index = low_trace.capture_prefix;
            return true;
        }
        std::size_t diff_index = low_trace.capture_prefix;
        if (!SnapshotEqual(low_trace, high_trace, i, &diff_index)) {
            *snapshot_index = i;
            *candidate_index = diff_index;
            return true;
        }
    }
    if (low_count != high_count) {
        *snapshot_index = common;
        *candidate_index = low_trace.capture_prefix;
        return true;
    }
    return false;
}

}  // namespace

int main() {
    constexpr std::size_t kNumNodes = 4096;
    constexpr std::size_t kDegree = 64;
    constexpr std::size_t kTopL = 2048;
    constexpr std::size_t kSearchWidth = 2;
    constexpr std::size_t kEntryCount = 128;
    constexpr std::size_t kMaxExpansions = 4096;
    constexpr std::size_t kCapturePrefix = kTopL;
    constexpr std::size_t kMaxSnapshots = 2100;
    constexpr std::size_t kPreview = 16;
    constexpr std::size_t kMaxSeeds = 64;

    const auto pivots_path =
        std::filesystem::current_path() / "topology_merge_diff_pivots.bin";
    const auto codes_path =
        std::filesystem::current_path() / "topology_merge_diff_codes.bin";

    topoanns::TopologySearchParams params;
    params.top_k = 10;
    params.top_l = kTopL;
    params.search_width = kSearchWidth;
    params.max_expansions = kMaxExpansions;

    DeviceTopologyDebugConfig debug_config;
    debug_config.max_snapshots = kMaxSnapshots;
    debug_config.capture_prefix = kCapturePrefix;

    for (std::uint32_t seed = 1; seed <= kMaxSeeds; ++seed) {
        WriteSyntheticPqFiles(pivots_path, codes_path, kNumNodes, seed * 17U + 3U);
        topoanns::SearchResources resources =
            topoanns::SearchResources::FromDataset(MakeRandomDataset(kNumNodes, kDegree, seed));
        resources.LoadPqIndex(pivots_path, codes_path);

        const std::vector<float> queries = {0.0f};
        const auto oracle = topoanns::PqDistanceOracle::FromFloatQueries(resources, queries, 1);

        std::vector<std::uint32_t> entries;
        entries.reserve(kEntryCount);
        for (std::size_t i = 0; i < kEntryCount; ++i) {
            entries.push_back(static_cast<std::uint32_t>(i));
        }
        auto entry_batch = MakeEntryBatch(entries);

        DeviceTopologyDebugTrace low_trace;
        unsetenv("TOPOANNS_ENABLE_HIGH_L_KERNEL");
        setenv("TOPOANNS_FORCE_LOW_L_KERNEL", "1", 1);
        auto low_result = topoanns::detail::RunTopologySearchKernelBatchDeviceDebug(
            resources, oracle, entry_batch, params, debug_config, &low_trace);
        const auto low_candidates = low_result.candidate_buffer.CopyToHost();

        DeviceTopologyDebugTrace high_trace;
        unsetenv("TOPOANNS_FORCE_LOW_L_KERNEL");
        setenv("TOPOANNS_ENABLE_HIGH_L_KERNEL", "1", 1);
        auto high_result = topoanns::detail::RunTopologySearchKernelBatchDeviceDebug(
            resources, oracle, entry_batch, params, debug_config, &high_trace);
        const auto high_candidates = high_result.candidate_buffer.CopyToHost();

        std::size_t final_diff = kTopL;
        for (std::size_t i = 0; i < kTopL; ++i) {
            const auto& lc = low_candidates[i];
            const auto& hc = high_candidates[i];
            if (lc.raw_node_id() != hc.raw_node_id() || lc.distance != hc.distance ||
                lc.expanded() != hc.expanded()) {
                final_diff = i;
                break;
            }
        }
        if (final_diff == kTopL) {
            continue;
        }

        std::size_t snapshot_index = 0;
        std::size_t candidate_index = 0;
        const bool found = FindFirstDiff(low_trace, high_trace, &snapshot_index, &candidate_index);

        std::cout << "seed=" << seed << " final_diff_index=" << final_diff << "\n";
        std::cout << "low_snapshot_count=" << low_trace.snapshot_counts[0]
                  << " high_snapshot_count=" << high_trace.snapshot_counts[0] << "\n";
        if (found && snapshot_index < low_trace.snapshot_counts[0] &&
            snapshot_index < high_trace.snapshot_counts[0]) {
            PrintSnapshotMetadata("low ", low_trace.snapshots[snapshot_index]);
            PrintSnapshotMetadata("high", high_trace.snapshots[snapshot_index]);
            std::cout << "first_diff_snapshot=" << snapshot_index
                      << " first_diff_candidate=" << candidate_index << "\n";
            const std::size_t base = snapshot_index * low_trace.capture_prefix;
            const std::size_t begin = candidate_index > 4 ? candidate_index - 4 : 0;
            const std::size_t end =
                std::min(low_trace.capture_prefix, candidate_index + kPreview);
            for (std::size_t i = begin; i < end; ++i) {
                PrintCandidate(low_trace.candidate_snapshots[base + i], i, "low ");
                PrintCandidate(high_trace.candidate_snapshots[base + i], i, "high");
            }
        } else {
            std::cout << "first_diff_snapshot=" << snapshot_index
                      << " due_to_snapshot_count_mismatch\n";
        }
        return 0;
    }

    std::cerr << "No low/high mismatch found in " << kMaxSeeds << " seeds." << std::endl;
    return 1;
}
