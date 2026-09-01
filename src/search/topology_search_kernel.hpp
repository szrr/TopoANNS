#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "topoanns/cuda_buffer.hpp"
#include "topoanns/device_entry_batch.hpp"
#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/topology_search.hpp"

namespace topoanns::detail {

constexpr std::uint32_t kExpandedNodeIdMask = 0x80000000U;
constexpr std::uint32_t kRawNodeIdMask = 0x7fffffffU;

__host__ __device__ inline bool PackedNodeIdValid(std::uint32_t packed_node_id) {
    return packed_node_id != kInvalidNodeId;
}

__host__ __device__ inline std::uint32_t RawNodeId(std::uint32_t packed_node_id) {
    return packed_node_id == kInvalidNodeId ? kInvalidNodeId : (packed_node_id & kRawNodeIdMask);
}

__host__ __device__ inline bool PackedNodeIdExpanded(std::uint32_t packed_node_id) {
    return packed_node_id != kInvalidNodeId && (packed_node_id & kExpandedNodeIdMask) != 0U;
}

__host__ __device__ inline std::uint32_t PackNodeId(std::uint32_t raw_node_id, bool expanded) {
    return raw_node_id == kInvalidNodeId
               ? kInvalidNodeId
               : (raw_node_id | (expanded ? kExpandedNodeIdMask : 0U));
}

struct alignas(8) DeviceTopologyCandidate {
    float distance = 0.0f;
    std::uint32_t node_id = kInvalidNodeId;

    __host__ __device__ bool valid() const { return PackedNodeIdValid(node_id); }

    __host__ __device__ std::uint32_t raw_node_id() const { return RawNodeId(node_id); }

    __host__ __device__ bool expanded() const { return PackedNodeIdExpanded(node_id); }

    __host__ __device__ void set_raw_node_id(std::uint32_t raw_node_id, bool expanded_flag = false) {
        node_id = PackNodeId(raw_node_id, expanded_flag);
    }

    __host__ __device__ void set_expanded(bool expanded_flag) {
        node_id = PackNodeId(raw_node_id(), expanded_flag);
    }
};

struct DeviceTopologySearchStats {
    std::uint32_t visited_nodes = 0;
    std::uint32_t expanded_nodes = 0;
    std::uint32_t topology_io_pages = 0;
    std::uint32_t iterations = 0;
    std::uint32_t valid_candidates = 0;
    std::uint32_t first_full_prefix_iteration = 0xffffffffU;
    std::uint32_t exact_reuse_inserts = 0;
    std::uint32_t exact_reuse_overflows = 0;
};

struct DeviceTopologyProfileCycles {
    unsigned long long pq_cycles = 0;
    unsigned long long pq_compute_cycles = 0;
    unsigned long long pq_prefetch_issue_cycles = 0;
    unsigned long long pq_prefetch_wait_cycles = 0;
    unsigned long long pq_checksum_cycles = 0;
    unsigned long long queue_cycles = 0;
    unsigned long long queue_scan_cycles = 0;
    unsigned long long queue_select_cycles = 0;
    unsigned long long frontier_sort_cycles = 0;
    unsigned long long tail_merge_cycles = 0;
    unsigned long long candidate_sort_cycles = 0;
    unsigned long long candidate_sort_before_full_prefix_cycles = 0;
    unsigned long long candidate_sort_after_full_prefix_cycles = 0;
    unsigned long long hash_rebuild_cycles = 0;
    unsigned long long learned_stop_model_cycles = 0;
    unsigned long long learned_stop_feature_cycles = 0;
    unsigned long long learned_stop_find_first_set_cycles = 0;
    unsigned long long learned_stop_count_bits_cycles = 0;
    unsigned long long learned_stop_topk_churn_cycles = 0;
    unsigned long long learned_stop_logit_eval_cycles = 0;
    unsigned long long combined_node_read_cycles = 0;
    unsigned long long exact_reuse_insert_cycles = 0;
};

struct DeviceTopologyDebugSnapshot {
    std::uint32_t merge_ordinal = 0;
    std::uint32_t phase = 0;
    std::uint32_t search_iteration = 0;
    std::uint32_t frontier_valid = 0;
    std::uint32_t accepted_frontier = 0;
    std::uint32_t selected_count = 0;
    std::uint32_t valid_candidates = 0;
    std::uint32_t visited_nodes = 0;
    std::uint32_t expanded_nodes = 0;
    std::uint64_t frontier_checksum = 0;
    std::uint64_t visited_hash_checksum = 0;
};

struct DeviceTopologyDebugConfig {
    std::uint32_t max_snapshots = 0;
    std::uint32_t capture_prefix = 0;
};

struct DeviceTopologyDebugTrace {
    std::size_t num_queries = 0;
    std::size_t max_snapshots = 0;
    std::size_t capture_prefix = 0;
    std::vector<std::uint32_t> snapshot_counts;
    std::vector<DeviceTopologyDebugSnapshot> snapshots;
    std::vector<DeviceTopologyCandidate> candidate_snapshots;
};

struct DeviceTopologyBatchResult {
    std::size_t num_queries = 0;
    std::size_t candidate_capacity = 0;
    CudaBuffer<DeviceTopologyCandidate> candidate_buffer;
    CudaBuffer<DeviceTopologySearchStats> stats_buffer;
    CudaBuffer<DeviceTopologyProfileCycles> profile_buffer;
    std::size_t expanded_trace_stride = 0;
    CudaBuffer<std::uint32_t> expanded_trace_buffer;
    CudaBuffer<std::uint32_t> gt_hit_mask_buffer;
    std::size_t exact_reuse_cache_capacity = 0;
    CudaBuffer<std::uint32_t> exact_reuse_node_ids;
    CudaBuffer<float> exact_reuse_distances;
    double kernel_ms = 0.0;
    std::size_t occupancy_dynamic_shared_bytes = 0;
    std::size_t occupancy_blocks_per_sm = 0;
    std::size_t occupancy_sm_count = 0;
    std::size_t occupancy_resident_blocks = 0;
    std::size_t occupancy_max_io_warps = 0;
};

struct TopologyMicrobatchExecutionConfig {
    std::size_t microbatch_queries = 2048;
    std::size_t context_depth = 3;
    std::size_t io_blocks = 10;
    std::size_t io_threads = 256;
};

struct TopologyMicrobatchExecutionProfile {
    double wall_ms = 0.0;
    double summed_io_kernel_ms = 0.0;
    std::uint64_t logical_io_requests = 0;
    std::size_t io_batches = 0;
    std::size_t nonempty_io_batches = 0;
    std::size_t min_nonempty_batch_requests = 0;
    std::size_t max_batch_requests = 0;
    std::uint64_t validation_mismatch_neighbors = 0;
};

DeviceTopologyBatchResult RunTopologySearchKernelBatchDeviceMicrobatched(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    const TopologyMicrobatchExecutionConfig& execution,
    TopologyMicrobatchExecutionProfile* profile);

DeviceTopologyBatchResult RunTopologySearchKernelBatchDevice(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    cudaStream_t stream);

using TopologyLaunchHostCallback = void (*)(void*);

DeviceTopologyBatchResult RunTopologySearchKernelBatchDeviceWithLaunchCallback(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    TopologyLaunchHostCallback launch_callback,
    void* launch_callback_context,
    cudaStream_t stream);

DeviceTopologyBatchResult RunTopologySearchKernelBatchDevice(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params);

DeviceTopologyBatchResult RunTopologySearchKernelBatchDevice(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const CudaBuffer<std::uint32_t>& gt_ids,
    std::uint32_t gt_topk,
    const TopologySearchParams& params);

DeviceTopologyBatchResult RunTopologySearchKernelBatchDeviceDebug(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    const DeviceTopologyDebugConfig& debug_config,
    DeviceTopologyDebugTrace* out_debug_trace);

std::vector<TopologySearchResult> RunTopologySearchKernelBatch(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const std::vector<std::vector<std::uint32_t>>& entries_by_query,
    std::size_t query_offset,
    std::size_t num_queries,
    const TopologySearchParams& params,
    TopologySearchBatchProfile* out_profile = nullptr);

std::vector<TopologySearchResult> RunTopologySearchKernelBatch(
    const SearchResources& resources,
    const PqDistanceOracle& distance_oracle,
    const DeviceEntryBatch& entry_batch,
    const TopologySearchParams& params,
    TopologySearchBatchProfile* out_profile = nullptr);

}  // namespace topoanns::detail
