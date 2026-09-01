#pragma once

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "topology_search_kernel.hpp"
#include "topoanns/pq_query_tables.hpp"
#include "topoanns/rerank_exact.hpp"
#include "topoanns/search_resources.hpp"

namespace topoanns::detail {

struct Pq2RefineBatchResult {
    std::size_t num_queries = 0;
    std::size_t candidate_capacity = 0;
    CudaBuffer<DeviceTopologyCandidate> candidate_buffer;
    PqQueryTablesProfile pq_profile;
    double kernel_ms = 0.0;
    std::size_t input_valid_candidates = 0;
    std::size_t input_valid_candidates_min = 0;
    std::size_t input_valid_candidates_max = 0;
    bool residual_refine = false;
};

struct PersistentBamRerankRunResult {
    CudaBuffer<float> final_distances;
    CudaBuffer<std::uint32_t> final_node_ids;
    std::size_t exact_count = 0;
    std::size_t rerank_ssd_io_pages = 0;
    std::size_t reused_exact_count = 0;
    std::size_t bound_filtered_count = 0;
    double kernel_ms = 0.0;
    double query_block_ms = 0.0;
    double exact_reuse_lookup_ms = 0.0;
    double learned_stop_model_ms = 0.0;
    double learned_stop_checkpoint_bookkeeping_ms = 0.0;
    double learned_stop_topk_churn_ms = 0.0;
    double learned_stop_next_window_scan_ms = 0.0;
    double learned_stop_logit_eval_ms = 0.0;
    std::size_t result_top_n = 0;
    std::size_t learned_stop_checkpoints = 0;
    std::size_t learned_stop_queries = 0;
    std::size_t learned_stop_prefix_sum = 0;
    std::size_t learned_stop_prefix_min = 0;
    std::size_t learned_stop_prefix_max = 0;
};

Pq2RefineBatchResult RunPq2RefineBatchDevice(const SearchResources& resources,
                                             const DeviceTopologyBatchResult& topology_result,
                                             const std::vector<float>& queries,
                                             std::size_t num_queries,
                                             std::size_t refine_top_l);

Pq2RefineBatchResult RunPq2RefineBatchDevice(const SearchResources& resources,
                                             const DeviceTopologyBatchResult& topology_result,
                                             const CudaBuffer<float>& device_queries,
                                             std::size_t num_queries,
                                             std::size_t refine_top_l,
                                             cudaStream_t stream);

Pq2RefineBatchResult RunHpqRefineBatchDevice(const SearchResources& resources,
                                             const DeviceTopologyBatchResult& topology_result,
                                             const CudaBuffer<float>& device_queries,
                                             std::size_t num_queries,
                                             std::size_t refine_top_l,
                                             cudaStream_t stream);

PersistentBamRerankRunResult RunPersistentBamRerankFloat32(
    const SearchResources& resources,
    const CudaBuffer<float>& device_queries,
    const DeviceTopologyBatchResult& topology_result,
    std::size_t num_queries,
    const RerankExactParams& params);

PersistentBamRerankRunResult RunPersistentBamRerankFloat32(
    const SearchResources& resources,
    const CudaBuffer<float>& device_queries,
    const std::vector<float>& host_queries,
    const DeviceTopologyBatchResult& topology_result,
    std::size_t num_queries,
    const RerankExactParams& params,
    cudaStream_t stream);

RerankBatchResult RunBatchFloat32FromDeviceTopology(
    const SearchResources& resources,
    const DeviceTopologyBatchResult& topology_result,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const RerankExactParams& params,
    RerankBatchProfile* out_profile = nullptr);

}  // namespace topoanns::detail
