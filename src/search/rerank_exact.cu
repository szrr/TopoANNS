#include "topoanns/rerank_exact.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "topoanns/bam_vector_provider.hpp"
#include "topoanns/cuda_buffer.hpp"
#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/vector_page_provider.hpp"
#include "fused_rerank_device.hpp"

namespace topoanns {
namespace {

constexpr unsigned int kFullMask = 0xffffffffU;
constexpr std::size_t kWarpSize = 32;
constexpr std::size_t kWarpsPerBlock = 4;
constexpr std::size_t kThreadsPerBlock = kWarpSize * kWarpsPerBlock;

bool DisableSpecializedExactDim768() {
    static const bool disabled = []() {
        const char* value = std::getenv("TOPOANNS_DISABLE_SPECIALIZED_EXACT_DIM768");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return disabled;
}

struct FlatRerankCandidate {
    std::uint32_t query_id = 0;
    std::uint32_t node_id = kInvalidNodeId;
    std::uint64_t page_id = 0;
    std::uint32_t slot_id = 0;
};

struct ExactDistanceRunResult {
    CudaBuffer<float> device_distances;
    double query_upload_ms = 0.0;
    double kernel_ms = 0.0;
};

constexpr std::size_t kMaxRerankSortCandidates = 2048;
constexpr std::size_t kRerankSortThreads = 256;
constexpr std::size_t kPq2RefineThreads = 256;
constexpr std::size_t kPq2RefineWarps = kPq2RefineThreads / kWarpSize;
// Residual PQ refine shares chunk layout with the base PQ. The MS-MARCO index
// uses 256-byte base PQ, so the rerank refine path must support 256 chunks.
constexpr std::size_t kMaxPq2RefineChunks = 256;
constexpr std::size_t kMaxHpqSelectorBytes = (kMaxPq2RefineChunks + 7U) / 8U;
constexpr std::size_t kMaxTrackedTopK = 128;
constexpr std::size_t kMaxPersistentTileSize = 128;
using host_cache_vec_t = ulonglong4;

void MergeMinMax(std::size_t src_min,
                 std::size_t src_max,
                 bool has_src,
                 std::size_t* dst_min,
                 std::size_t* dst_max,
                 bool* has_dst) {
    if (!has_src) {
        return;
    }
    if (!*has_dst) {
        *dst_min = src_min;
        *dst_max = src_max;
        *has_dst = true;
        return;
    }
    *dst_min = std::min(*dst_min, src_min);
    *dst_max = std::max(*dst_max, src_max);
}

std::size_t NextPowerOfTwo(std::size_t value) {
    std::size_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

__device__ inline bool RerankLess(float lhs_distance,
                                  std::uint32_t lhs_node_id,
                                  float rhs_distance,
                                  std::uint32_t rhs_node_id) {
    if (lhs_distance != rhs_distance) {
        return lhs_distance < rhs_distance;
    }
    return lhs_node_id < rhs_node_id;
}

__device__ __forceinline__ std::uint32_t GetWarpOid(std::uint32_t eq_mask) {
    const std::uint32_t lane = static_cast<std::uint32_t>(threadIdx.x) & (kWarpSize - 1U);
    const std::uint32_t lower_mask = lane == 0U ? 0U : ((1U << lane) - 1U);
    return __popc(eq_mask & lower_mask);
}

__device__ __forceinline__ int copyFromHostCacheVec(const void* data_in,
                                                    void* data_out,
                                                    std::size_t bytes,
                                                    std::uint32_t eq_mask) {
    const std::uint32_t num_threads = __popc(eq_mask);
    const std::uint32_t oid = GetWarpOid(eq_mask);
    const std::uintptr_t combined_alignment =
        reinterpret_cast<std::uintptr_t>(data_in) |
        reinterpret_cast<std::uintptr_t>(data_out) |
        static_cast<std::uintptr_t>(bytes);

    if ((combined_alignment & (alignof(host_cache_vec_t) - 1U)) == 0U) {
        const auto* data_in_vec = reinterpret_cast<const host_cache_vec_t*>(data_in);
        auto* data_out_vec = reinterpret_cast<host_cache_vec_t*>(data_out);
        const std::size_t vec_count = bytes / sizeof(host_cache_vec_t);
        for (std::size_t i = oid; i < vec_count; i += num_threads) {
            data_out_vec[i] = data_in_vec[i];
        }
        return static_cast<int>(bytes);
    }

    const auto* data_in_byte = reinterpret_cast<const std::uint8_t*>(data_in);
    auto* data_out_byte = reinterpret_cast<std::uint8_t*>(data_out);
    for (std::size_t i = oid; i < bytes; i += num_threads) {
        data_out_byte[i] = data_in_byte[i];
    }
    return static_cast<int>(bytes);
}

__device__ __forceinline__ float SquaredL2LowerBound(float approx_sq_distance,
                                                     float error_bound_l2) {
    const float approx_l2 = sqrtf(fmaxf(approx_sq_distance, 0.0f));
    const float lower_l2 = fmaxf(0.0f, approx_l2 - error_bound_l2);
    return lower_l2 * lower_l2;
}

template <typename T>
__global__ void fill_buffer_kernel(T* data, std::size_t count, T value) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) {
        data[idx] = value;
    }
}

__global__ void rerank_bitonic_sort_kernel(float* distances,
                                           std::uint32_t* node_ids,
                                           const std::uint32_t* query_offsets,
                                           std::size_t padded_count) {
    const std::size_t query_id = blockIdx.x;
    const std::size_t query_begin = query_offsets[query_id];
    const std::size_t query_end = query_offsets[query_id + 1];
    const std::size_t candidate_count = query_end - query_begin;

    __shared__ float shared_distances[kMaxRerankSortCandidates];
    __shared__ std::uint32_t shared_node_ids[kMaxRerankSortCandidates];

    for (std::size_t idx = threadIdx.x; idx < padded_count; idx += blockDim.x) {
        if (idx < candidate_count) {
            shared_distances[idx] = distances[query_begin + idx];
            shared_node_ids[idx] = node_ids[query_begin + idx];
        } else {
            shared_distances[idx] = std::numeric_limits<float>::infinity();
            shared_node_ids[idx] = kInvalidNodeId;
        }
    }
    __syncthreads();

    for (std::size_t size = 2; size <= padded_count; size <<= 1) {
        for (std::size_t stride = size >> 1; stride > 0; stride >>= 1) {
            for (std::size_t idx = threadIdx.x; idx < padded_count; idx += blockDim.x) {
                const std::size_t partner = idx ^ stride;
                if (partner <= idx || partner >= padded_count) {
                    continue;
                }
                const bool ascending = (idx & size) == 0;
                const bool partner_is_smaller =
                    RerankLess(shared_distances[partner], shared_node_ids[partner],
                               shared_distances[idx], shared_node_ids[idx]);
                if (partner_is_smaller == ascending) {
                    const float tmp_distance = shared_distances[idx];
                    const std::uint32_t tmp_node_id = shared_node_ids[idx];
                    shared_distances[idx] = shared_distances[partner];
                    shared_node_ids[idx] = shared_node_ids[partner];
                    shared_distances[partner] = tmp_distance;
                    shared_node_ids[partner] = tmp_node_id;
                }
            }
            __syncthreads();
        }
    }

    for (std::size_t idx = threadIdx.x; idx < candidate_count; idx += blockDim.x) {
        distances[query_begin + idx] = shared_distances[idx];
        node_ids[query_begin + idx] = shared_node_ids[idx];
    }
}

template <typename T>
std::vector<T> UnpackTypedVectors(const std::vector<std::uint8_t>& bytes,
                                  std::size_t num_vectors,
                                  std::size_t dim) {
    std::vector<T> unpacked(num_vectors * dim);
    if (!bytes.empty()) {
        std::memcpy(unpacked.data(), bytes.data(), bytes.size());
    }
    return unpacked;
}

template <typename T>
__global__ void unpack_candidate_vectors_kernel(const std::uint8_t* page_bytes,
                                                std::size_t page_size_bytes,
                                                const std::uint32_t* page_indices,
                                                const std::uint32_t* slot_ids,
                                                std::size_t vector_bytes,
                                                std::size_t dim,
                                                std::size_t num_candidates,
                                                T* out_vectors) {
    const std::size_t total_elements = num_candidates * dim;
    const std::size_t thread_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t element_id = thread_id; element_id < total_elements; element_id += stride) {
        const std::size_t candidate_id = element_id / dim;
        const std::size_t dim_id = element_id % dim;
        const std::size_t source_offset =
            static_cast<std::size_t>(page_indices[candidate_id]) * page_size_bytes +
            static_cast<std::size_t>(slot_ids[candidate_id]) * vector_bytes +
            dim_id * sizeof(T);
        out_vectors[element_id] =
            *reinterpret_cast<const T*>(page_bytes + source_offset);
    }
}

template <typename QueryT, typename VectorT>
__global__ void rerank_exact_kernel(const QueryT* queries,
                                    const VectorT* candidate_vectors,
                                    const std::uint32_t* candidate_query_ids,
                                    std::size_t dim,
                                    std::size_t num_candidates,
                                    float* out_distances) {
    const std::size_t warp_global_id =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    if (warp_global_id >= num_candidates) {
        return;
    }

    const std::uint32_t query_id = candidate_query_ids[warp_global_id];
    const QueryT* query = queries + static_cast<std::size_t>(query_id) * dim;
    const VectorT* vector = candidate_vectors + warp_global_id * dim;

    float distance = 0.0f;
    for (std::size_t dim_idx = lane_id; dim_idx < dim; dim_idx += kWarpSize) {
        const float diff =
            static_cast<float>(vector[dim_idx]) - static_cast<float>(query[dim_idx]);
        distance += diff * diff;
    }

    distance += __shfl_down_sync(kFullMask, distance, 16);
    distance += __shfl_down_sync(kFullMask, distance, 8);
    distance += __shfl_down_sync(kFullMask, distance, 4);
    distance += __shfl_down_sync(kFullMask, distance, 2);
    distance += __shfl_down_sync(kFullMask, distance, 1);

    if (lane_id == 0) {
        out_distances[warp_global_id] = distance;
    }
}

template <std::size_t kDim>
__global__ void rerank_exact_kernel_float32_dim(const float* queries,
                                                const float* candidate_vectors,
                                                const std::uint32_t* candidate_query_ids,
                                                std::size_t num_candidates,
                                                float* out_distances) {
    const std::size_t warp_global_id =
        (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarpSize;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    if (warp_global_id >= num_candidates) {
        return;
    }

    const std::uint32_t query_id = candidate_query_ids[warp_global_id];
    const float* query = queries + static_cast<std::size_t>(query_id) * kDim;
    const float* vector = candidate_vectors + warp_global_id * kDim;

    float distance = 0.0f;
#pragma unroll
    for (std::size_t dim_idx = lane_id; dim_idx < kDim; dim_idx += kWarpSize) {
        const float diff = vector[dim_idx] - query[dim_idx];
        distance += diff * diff;
    }

    distance += __shfl_down_sync(kFullMask, distance, 16);
    distance += __shfl_down_sync(kFullMask, distance, 8);
    distance += __shfl_down_sync(kFullMask, distance, 4);
    distance += __shfl_down_sync(kFullMask, distance, 2);
    distance += __shfl_down_sync(kFullMask, distance, 1);

    if (lane_id == 0) {
        out_distances[warp_global_id] = distance;
    }
}

inline void LaunchRerankExactKernelFloat32Specialized(
    const float* queries,
    const float* candidate_vectors,
    const std::uint32_t* candidate_query_ids,
    std::size_t dim,
    std::size_t num_candidates,
    float* out_distances,
    std::size_t num_blocks) {
    switch (dim) {
        case 128:
            rerank_exact_kernel_float32_dim<128><<<num_blocks, kThreadsPerBlock>>>(
                queries, candidate_vectors, candidate_query_ids, num_candidates, out_distances);
            break;
        case 768:
            if (DisableSpecializedExactDim768()) {
                rerank_exact_kernel<<<num_blocks, kThreadsPerBlock>>>(
                    queries, candidate_vectors, candidate_query_ids, dim, num_candidates,
                    out_distances);
            } else {
                rerank_exact_kernel_float32_dim<768><<<num_blocks, kThreadsPerBlock>>>(
                    queries, candidate_vectors, candidate_query_ids, num_candidates, out_distances);
            }
            break;
        default:
            rerank_exact_kernel<<<num_blocks, kThreadsPerBlock>>>(
                queries, candidate_vectors, candidate_query_ids, dim, num_candidates,
                out_distances);
            break;
    }
}

__global__ void prepare_topology_candidates_for_fused_rerank_kernel(
    const detail::DeviceTopologyCandidate* topology_candidates,
    std::size_t candidate_capacity,
    std::size_t top_n,
    std::size_t vectors_per_page,
    std::uint64_t* out_page_ids,
    std::uint32_t* out_slot_ids,
    std::uint32_t* out_node_ids,
    std::uint32_t* out_query_ids,
    std::uint32_t* out_valid_count) {
    const std::size_t query_id = blockIdx.x;
    const std::size_t base = query_id * top_n;
    std::uint32_t local_valid = 0;
    for (std::size_t rank = threadIdx.x; rank < top_n; rank += blockDim.x) {
        const std::size_t candidate_index = query_id * candidate_capacity + rank;
        const detail::DeviceTopologyCandidate candidate = topology_candidates[candidate_index];
        out_query_ids[base + rank] = static_cast<std::uint32_t>(query_id);
        if (candidate.valid()) {
            const std::uint32_t raw_node_id = candidate.raw_node_id();
            out_node_ids[base + rank] = raw_node_id;
            out_page_ids[base + rank] =
                static_cast<std::uint64_t>(raw_node_id) / vectors_per_page;
            out_slot_ids[base + rank] =
                static_cast<std::uint32_t>(raw_node_id % vectors_per_page);
            ++local_valid;
        } else {
            out_node_ids[base + rank] = kInvalidNodeId;
            out_page_ids[base + rank] = 0;
            out_slot_ids[base + rank] = 0;
        }
    }

    __shared__ std::uint32_t block_valid;
    if (threadIdx.x == 0) {
        block_valid = 0;
    }
    __syncthreads();
    atomicAdd(&block_valid, local_valid);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(out_valid_count, block_valid);
    }
}

__global__ void prepare_topology_candidates_for_rank_tile_kernel(
    const detail::DeviceTopologyCandidate* topology_candidates,
    std::size_t candidate_capacity,
    std::size_t rank_begin,
    std::size_t rank_count,
    std::size_t vectors_per_page,
    std::uint64_t* out_page_ids,
    std::uint32_t* out_slot_ids,
    std::uint32_t* out_node_ids,
    std::uint32_t* out_query_ids,
    std::uint32_t* out_valid_count) {
    const std::size_t query_id = blockIdx.x;
    const std::size_t base = query_id * rank_count;
    std::uint32_t local_valid = 0;
    for (std::size_t local_rank = threadIdx.x; local_rank < rank_count; local_rank += blockDim.x) {
        const std::size_t candidate_rank = rank_begin + local_rank;
        const std::size_t candidate_index = query_id * candidate_capacity + candidate_rank;
        const detail::DeviceTopologyCandidate candidate = topology_candidates[candidate_index];
        out_query_ids[base + local_rank] = static_cast<std::uint32_t>(query_id);
        if (candidate.valid()) {
            const std::uint32_t raw_node_id = candidate.raw_node_id();
            out_node_ids[base + local_rank] = raw_node_id;
            out_page_ids[base + local_rank] =
                static_cast<std::uint64_t>(raw_node_id) / vectors_per_page;
            out_slot_ids[base + local_rank] =
                static_cast<std::uint32_t>(raw_node_id % vectors_per_page);
            ++local_valid;
        } else {
            out_node_ids[base + local_rank] = kInvalidNodeId;
            out_page_ids[base + local_rank] = 0;
            out_slot_ids[base + local_rank] = 0;
        }
    }

    __shared__ std::uint32_t block_valid;
    if (threadIdx.x == 0) {
        block_valid = 0;
    }
    __syncthreads();
    atomicAdd(&block_valid, local_valid);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(out_valid_count, block_valid);
    }
}

__global__ void prepare_topology_candidates_for_rank_tile_compact_with_bound_kernel(
    const detail::DeviceTopologyCandidate* topology_candidates,
    std::size_t candidate_capacity,
    std::size_t rank_begin,
    std::size_t rank_count,
    std::size_t vectors_per_page,
    const float* error_bounds,
    const float* query_norm_squares,
    const float* query_thresholds,
    std::uint64_t* out_page_ids,
    std::uint32_t* out_slot_ids,
    std::uint32_t* out_node_ids,
    std::uint32_t* out_query_ids,
    std::uint32_t* out_local_ranks,
    std::uint32_t* out_valid_count,
    std::uint32_t* out_filtered_count) {
    const std::size_t query_id = blockIdx.x;
    const float query_threshold = query_thresholds[query_id];
    const bool threshold_ready = isfinite(query_threshold);
    std::uint32_t local_filtered = 0;

    for (std::size_t local_rank = threadIdx.x; local_rank < rank_count; local_rank += blockDim.x) {
        const std::size_t candidate_rank = rank_begin + local_rank;
        const std::size_t candidate_index = query_id * candidate_capacity + candidate_rank;
        const detail::DeviceTopologyCandidate candidate = topology_candidates[candidate_index];
        if (!candidate.valid()) {
            continue;
        }
        const std::uint32_t raw_node_id = candidate.raw_node_id();

        bool keep = true;
        if (threshold_ready) {
            const float approx_sq_distance =
                candidate.distance - query_norm_squares[query_id];
            const float lower_bound =
                SquaredL2LowerBound(approx_sq_distance, error_bounds[raw_node_id]);
            if (lower_bound > query_threshold) {
                keep = false;
                ++local_filtered;
            }
        }
        if (!keep) {
            continue;
        }

        const std::uint32_t slot = atomicAdd(out_valid_count, 1U);
        out_page_ids[slot] = static_cast<std::uint64_t>(raw_node_id) / vectors_per_page;
        out_slot_ids[slot] = static_cast<std::uint32_t>(raw_node_id % vectors_per_page);
        out_node_ids[slot] = raw_node_id;
        out_query_ids[slot] = static_cast<std::uint32_t>(query_id);
        out_local_ranks[slot] = static_cast<std::uint32_t>(local_rank);
    }

    __shared__ std::uint32_t block_filtered;
    if (threadIdx.x == 0) {
        block_filtered = 0;
    }
    __syncthreads();
    atomicAdd(&block_filtered, local_filtered);
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(out_filtered_count, block_filtered);
    }
}

__global__ void scatter_rank_tile_results_kernel(const float* tile_distances,
                                                 const std::uint32_t* tile_node_ids,
                                                 std::size_t top_n,
                                                 std::size_t rank_begin,
                                                 std::size_t rank_count,
                                                 std::size_t tile_count,
                                                 float* out_distances,
                                                 std::uint32_t* out_node_ids) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= tile_count) {
        return;
    }
    const std::size_t query_id = idx / rank_count;
    const std::size_t local_rank = idx % rank_count;
    const std::size_t global_rank = rank_begin + local_rank;
    const std::size_t global_index = query_id * top_n + global_rank;
    out_distances[global_index] = tile_distances[idx];
    out_node_ids[global_index] = tile_node_ids[idx];
}

__global__ void scatter_rank_tile_compact_results_kernel(const float* tile_distances,
                                                         const std::uint32_t* tile_node_ids,
                                                         const std::uint32_t* tile_query_ids,
                                                         const std::uint32_t* tile_local_ranks,
                                                         std::size_t top_n,
                                                         std::size_t rank_begin,
                                                         std::size_t valid_count,
                                                         float* out_distances,
                                                         std::uint32_t* out_node_ids) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= valid_count) {
        return;
    }
    const std::size_t query_id = tile_query_ids[idx];
    const std::size_t global_rank = rank_begin + tile_local_ranks[idx];
    const std::size_t global_index = query_id * top_n + global_rank;
    out_distances[global_index] = tile_distances[idx];
    out_node_ids[global_index] = tile_node_ids[idx];
}

__global__ void update_query_thresholds_from_final_distances_kernel(
    const float* final_distances,
    std::size_t top_n,
    std::size_t prefix_count,
    std::size_t top_k,
    float* out_thresholds) {
    const std::size_t query_id = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    float best[kMaxTrackedTopK];
    for (std::size_t i = 0; i < top_k; ++i) {
        best[i] = std::numeric_limits<float>::infinity();
    }

    const float* query_distances = final_distances + query_id * top_n;
    for (std::size_t rank = 0; rank < prefix_count; ++rank) {
        const float distance = query_distances[rank];
        if (!(distance < best[top_k - 1])) {
            continue;
        }
        std::size_t insert = top_k - 1;
        while (insert > 0 && distance < best[insert - 1]) {
            best[insert] = best[insert - 1];
            --insert;
        }
        best[insert] = distance;
    }
    out_thresholds[query_id] = best[top_k - 1];
}

__global__ void pq2_refine_bitonic_sort_kernel(
    const detail::DeviceTopologyCandidate* topology_candidates,
    const detail::DeviceTopologySearchStats* topology_stats,
    std::size_t candidate_capacity,
    std::size_t refine_top_l,
    std::size_t padded_top_l,
    const std::uint8_t* pq2_codes,
    std::size_t num_chunks,
    const float* query_tables,
    std::size_t query_stride,
    detail::DeviceTopologyCandidate* out_candidates) {
    const std::size_t query_id = blockIdx.x;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    const std::size_t query_candidate_base = query_id * candidate_capacity;
    const std::size_t out_base = query_id * refine_top_l;
    const std::uint32_t active_mask = __activemask();

    __shared__ float shared_distances[kMaxRerankSortCandidates];
    __shared__ std::uint32_t shared_node_ids[kMaxRerankSortCandidates];
    __shared__ std::uint8_t shared_codes[kPq2RefineWarps][kMaxPq2RefineChunks];

    const std::size_t valid_count =
        min(static_cast<std::size_t>(topology_stats[query_id].valid_candidates), refine_top_l);
    const float* query_table = query_tables + query_id * query_stride;

    for (std::size_t rank = warp_id; rank < refine_top_l; rank += kPq2RefineWarps) {
        std::uint32_t node_id = kInvalidNodeId;
        float distance = std::numeric_limits<float>::infinity();
        if (rank < valid_count) {
            node_id = topology_candidates[query_candidate_base + rank].raw_node_id();
            if (node_id != kInvalidNodeId) {
                copyFromHostCacheVec(pq2_codes + static_cast<std::size_t>(node_id) * num_chunks,
                                     shared_codes[warp_id], num_chunks, active_mask);
                __syncwarp(active_mask);
                float partial = 0.0f;
                for (std::size_t chunk = lane_id; chunk < num_chunks; chunk += kWarpSize) {
                    partial +=
                        query_table[chunk * kNumPqCentroids + shared_codes[warp_id][chunk]];
                }
                partial += __shfl_down_sync(kFullMask, partial, 16);
                partial += __shfl_down_sync(kFullMask, partial, 8);
                partial += __shfl_down_sync(kFullMask, partial, 4);
                partial += __shfl_down_sync(kFullMask, partial, 2);
                partial += __shfl_down_sync(kFullMask, partial, 1);
                if (lane_id == 0) {
                    distance = partial;
                }
            }
        }
        if (lane_id == 0) {
            shared_distances[rank] = distance;
            shared_node_ids[rank] = node_id;
        }
    }

    for (std::size_t idx = threadIdx.x + refine_top_l; idx < padded_top_l; idx += blockDim.x) {
        shared_distances[idx] = std::numeric_limits<float>::infinity();
        shared_node_ids[idx] = kInvalidNodeId;
    }
    __syncthreads();

    for (std::size_t size = 2; size <= padded_top_l; size <<= 1) {
        for (std::size_t stride = size >> 1; stride > 0; stride >>= 1) {
            for (std::size_t idx = threadIdx.x; idx < padded_top_l; idx += blockDim.x) {
                const std::size_t partner = idx ^ stride;
                if (partner <= idx || partner >= padded_top_l) {
                    continue;
                }
                const bool ascending = (idx & size) == 0;
                const bool partner_is_smaller =
                    RerankLess(shared_distances[partner], shared_node_ids[partner],
                               shared_distances[idx], shared_node_ids[idx]);
                if (partner_is_smaller == ascending) {
                    const float tmp_distance = shared_distances[idx];
                    const std::uint32_t tmp_node_id = shared_node_ids[idx];
                    shared_distances[idx] = shared_distances[partner];
                    shared_node_ids[idx] = shared_node_ids[partner];
                    shared_distances[partner] = tmp_distance;
                    shared_node_ids[partner] = tmp_node_id;
                }
            }
            __syncthreads();
        }
    }

    for (std::size_t idx = threadIdx.x; idx < refine_top_l; idx += blockDim.x) {
        detail::DeviceTopologyCandidate candidate;
        candidate.distance = shared_distances[idx];
        candidate.set_raw_node_id(shared_node_ids[idx], false);
        out_candidates[out_base + idx] = candidate;
    }
}

__global__ void hpq_refine_bitonic_sort_kernel(
    const detail::DeviceTopologyCandidate* topology_candidates,
    const detail::DeviceTopologySearchStats* topology_stats,
    std::size_t candidate_capacity,
    std::size_t refine_top_l,
    std::size_t padded_top_l,
    const std::uint8_t* hybrid_codes,
    const std::uint8_t* selector_bits,
    std::size_t selector_stride_bytes,
    std::size_t num_chunks,
    const float* base_query_tables,
    const float* outlier_query_tables,
    std::size_t query_stride,
    detail::DeviceTopologyCandidate* out_candidates) {
    const std::size_t query_id = blockIdx.x;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    const std::size_t query_candidate_base = query_id * candidate_capacity;
    const std::size_t out_base = query_id * refine_top_l;
    const std::uint32_t active_mask = __activemask();

    __shared__ float shared_distances[kMaxRerankSortCandidates];
    __shared__ std::uint32_t shared_node_ids[kMaxRerankSortCandidates];
    __shared__ std::uint8_t shared_codes[kPq2RefineWarps][kMaxPq2RefineChunks];
    __shared__ std::uint8_t shared_selectors[kPq2RefineWarps][kMaxHpqSelectorBytes];

    const std::size_t valid_count =
        min(static_cast<std::size_t>(topology_stats[query_id].valid_candidates), refine_top_l);
    const float* base_query_table = base_query_tables + query_id * query_stride;
    const float* outlier_query_table = outlier_query_tables + query_id * query_stride;

    for (std::size_t rank = warp_id; rank < refine_top_l; rank += kPq2RefineWarps) {
        std::uint32_t node_id = kInvalidNodeId;
        float distance = std::numeric_limits<float>::infinity();
        if (rank < valid_count) {
            node_id = topology_candidates[query_candidate_base + rank].raw_node_id();
            if (node_id != kInvalidNodeId) {
                copyFromHostCacheVec(
                    hybrid_codes + static_cast<std::size_t>(node_id) * num_chunks,
                    shared_codes[warp_id], num_chunks, active_mask);
                copyFromHostCacheVec(
                    selector_bits + static_cast<std::size_t>(node_id) * selector_stride_bytes,
                    shared_selectors[warp_id], selector_stride_bytes, active_mask);
                __syncwarp(active_mask);
                float partial = 0.0f;
                for (std::size_t chunk = lane_id; chunk < num_chunks; chunk += kWarpSize) {
                    const bool use_outlier =
                        ((shared_selectors[warp_id][chunk >> 3U] >> (chunk & 7U)) & 1U) != 0;
                    const std::uint8_t code = shared_codes[warp_id][chunk];
                    const float* table = use_outlier ? outlier_query_table : base_query_table;
                    partial += table[chunk * kNumPqCentroids + code];
                }
                partial += __shfl_down_sync(kFullMask, partial, 16);
                partial += __shfl_down_sync(kFullMask, partial, 8);
                partial += __shfl_down_sync(kFullMask, partial, 4);
                partial += __shfl_down_sync(kFullMask, partial, 2);
                partial += __shfl_down_sync(kFullMask, partial, 1);
                if (lane_id == 0) {
                    distance = partial;
                }
            }
        }
        if (lane_id == 0) {
            shared_distances[rank] = distance;
            shared_node_ids[rank] = node_id;
        }
    }

    for (std::size_t idx = threadIdx.x + refine_top_l; idx < padded_top_l; idx += blockDim.x) {
        shared_distances[idx] = std::numeric_limits<float>::infinity();
        shared_node_ids[idx] = kInvalidNodeId;
    }
    __syncthreads();

    for (std::size_t size = 2; size <= padded_top_l; size <<= 1) {
        for (std::size_t stride = size >> 1; stride > 0; stride >>= 1) {
            for (std::size_t idx = threadIdx.x; idx < padded_top_l; idx += blockDim.x) {
                const std::size_t partner = idx ^ stride;
                if (partner <= idx || partner >= padded_top_l) {
                    continue;
                }
                const bool ascending = (idx & size) == 0;
                const bool partner_is_smaller =
                    RerankLess(shared_distances[partner], shared_node_ids[partner],
                               shared_distances[idx], shared_node_ids[idx]);
                if (partner_is_smaller == ascending) {
                    const float tmp_distance = shared_distances[idx];
                    const std::uint32_t tmp_node_id = shared_node_ids[idx];
                    shared_distances[idx] = shared_distances[partner];
                    shared_node_ids[idx] = shared_node_ids[partner];
                    shared_distances[partner] = tmp_distance;
                    shared_node_ids[partner] = tmp_node_id;
                }
            }
            __syncthreads();
        }
    }

    for (std::size_t idx = threadIdx.x; idx < refine_top_l; idx += blockDim.x) {
        detail::DeviceTopologyCandidate candidate;
        candidate.distance = shared_distances[idx];
        candidate.set_raw_node_id(shared_node_ids[idx], false);
        out_candidates[out_base + idx] = candidate;
    }
}

__global__ void pq2_residual_refine_bitonic_sort_kernel(
    const detail::DeviceTopologyCandidate* topology_candidates,
    const detail::DeviceTopologySearchStats* topology_stats,
    std::size_t candidate_capacity,
    std::size_t refine_top_l,
    std::size_t padded_top_l,
    const std::uint8_t* base_codes,
    const std::uint8_t* residual_codes,
    std::size_t num_chunks,
    const float* residual_query_tables,
    std::size_t query_stride,
    const float* cross_terms,
    detail::DeviceTopologyCandidate* out_candidates) {
    const std::size_t query_id = blockIdx.x;
    const std::size_t lane_id = threadIdx.x % kWarpSize;
    const std::size_t warp_id = threadIdx.x / kWarpSize;
    const std::size_t query_candidate_base = query_id * candidate_capacity;
    const std::size_t out_base = query_id * refine_top_l;
    const std::uint32_t active_mask = __activemask();

    __shared__ float shared_distances[kMaxRerankSortCandidates];
    __shared__ std::uint32_t shared_node_ids[kMaxRerankSortCandidates];
    __shared__ std::uint8_t shared_base_codes[kPq2RefineWarps][kMaxPq2RefineChunks];
    __shared__ std::uint8_t shared_residual_codes[kPq2RefineWarps][kMaxPq2RefineChunks];

    const std::size_t valid_count =
        min(static_cast<std::size_t>(topology_stats[query_id].valid_candidates), refine_top_l);
    const float* residual_query_table = residual_query_tables + query_id * query_stride;

    for (std::size_t rank = warp_id; rank < refine_top_l; rank += kPq2RefineWarps) {
        std::uint32_t node_id = kInvalidNodeId;
        float distance = std::numeric_limits<float>::infinity();
        if (rank < valid_count) {
            const detail::DeviceTopologyCandidate candidate =
                topology_candidates[query_candidate_base + rank];
            node_id = candidate.raw_node_id();
            if (node_id != kInvalidNodeId) {
                const std::size_t code_offset = static_cast<std::size_t>(node_id) * num_chunks;
                copyFromHostCacheVec(residual_codes + code_offset, shared_residual_codes[warp_id],
                                     num_chunks, active_mask);
                for (std::size_t chunk = lane_id; chunk < num_chunks; chunk += kWarpSize) {
                    shared_base_codes[warp_id][chunk] = base_codes[code_offset + chunk];
                }
                __syncwarp(active_mask);

                float partial = 0.0f;
                for (std::size_t chunk = lane_id; chunk < num_chunks; chunk += kWarpSize) {
                    const std::uint32_t base_center = shared_base_codes[warp_id][chunk];
                    const std::uint32_t residual_center = shared_residual_codes[warp_id][chunk];
                    partial +=
                        residual_query_table[chunk * kNumPqCentroids + residual_center] +
                        cross_terms[(chunk * kNumPqCentroids + base_center) * kNumPqCentroids +
                                    residual_center];
                }
                partial += __shfl_down_sync(kFullMask, partial, 16);
                partial += __shfl_down_sync(kFullMask, partial, 8);
                partial += __shfl_down_sync(kFullMask, partial, 4);
                partial += __shfl_down_sync(kFullMask, partial, 2);
                partial += __shfl_down_sync(kFullMask, partial, 1);
                if (lane_id == 0) {
                    distance = candidate.distance + partial;
                }
            }
        }
        if (lane_id == 0) {
            shared_distances[rank] = distance;
            shared_node_ids[rank] = node_id;
        }
    }

    for (std::size_t idx = threadIdx.x + refine_top_l; idx < padded_top_l; idx += blockDim.x) {
        shared_distances[idx] = std::numeric_limits<float>::infinity();
        shared_node_ids[idx] = kInvalidNodeId;
    }
    __syncthreads();

    for (std::size_t size = 2; size <= padded_top_l; size <<= 1) {
        for (std::size_t stride = size >> 1; stride > 0; stride >>= 1) {
            for (std::size_t idx = threadIdx.x; idx < padded_top_l; idx += blockDim.x) {
                const std::size_t partner = idx ^ stride;
                if (partner <= idx || partner >= padded_top_l) {
                    continue;
                }
                const bool ascending = (idx & size) == 0;
                const bool partner_is_smaller =
                    RerankLess(shared_distances[partner], shared_node_ids[partner],
                               shared_distances[idx], shared_node_ids[idx]);
                if (partner_is_smaller == ascending) {
                    const float tmp_distance = shared_distances[idx];
                    const std::uint32_t tmp_node_id = shared_node_ids[idx];
                    shared_distances[idx] = shared_distances[partner];
                    shared_node_ids[idx] = shared_node_ids[partner];
                    shared_distances[partner] = tmp_distance;
                    shared_node_ids[partner] = tmp_node_id;
                }
            }
            __syncthreads();
        }
    }

    for (std::size_t idx = threadIdx.x; idx < refine_top_l; idx += blockDim.x) {
        detail::DeviceTopologyCandidate candidate;
        candidate.distance = shared_distances[idx];
        candidate.set_raw_node_id(shared_node_ids[idx], false);
        out_candidates[out_base + idx] = candidate;
    }
}

template <typename T>
__global__ void unpack_candidate_vectors_linear_pages_kernel(const std::uint8_t* page_bytes,
                                                             std::size_t page_size_bytes,
                                                             const std::uint32_t* slot_ids,
                                                             std::size_t vector_bytes,
                                                             std::size_t dim,
                                                             std::size_t num_candidates,
                                                             T* out_vectors) {
    const std::size_t total_elements = num_candidates * dim;
    const std::size_t thread_id =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t element_id = thread_id; element_id < total_elements; element_id += stride) {
        const std::size_t candidate_id = element_id / dim;
        const std::size_t dim_id = element_id % dim;
        const std::size_t source_offset =
            candidate_id * page_size_bytes +
            static_cast<std::size_t>(slot_ids[candidate_id]) * vector_bytes +
            dim_id * sizeof(T);
        out_vectors[element_id] =
            *reinterpret_cast<const T*>(page_bytes + source_offset);
    }
}

__global__ void mask_invalid_rerank_distances_kernel(const std::uint32_t* node_ids,
                                                     std::size_t count,
                                                     float* distances) {
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count) {
        return;
    }
    if (node_ids[idx] == kInvalidNodeId) {
        distances[idx] = std::numeric_limits<float>::infinity();
    }
}

template <typename QueryT, typename VectorT>
ExactDistanceRunResult ComputeExactDistances(
    const std::vector<QueryT>& queries,
    const std::vector<std::uint8_t>& candidate_vector_bytes,
    const std::vector<std::uint32_t>& candidate_query_ids,
    std::size_t dim) {
    const std::size_t num_candidates = candidate_query_ids.size();
    if (num_candidates == 0) {
        return ExactDistanceRunResult{};
    }

    const std::vector<VectorT> candidate_vectors =
        UnpackTypedVectors<VectorT>(candidate_vector_bytes, num_candidates, dim);
    const auto upload_begin = std::chrono::steady_clock::now();
    CudaBuffer<QueryT> query_buffer = CudaBuffer<QueryT>::CopyFromHost(queries);
    const auto upload_end = std::chrono::steady_clock::now();
    CudaBuffer<VectorT> vector_buffer = CudaBuffer<VectorT>::CopyFromHost(candidate_vectors);
    CudaBuffer<std::uint32_t> query_id_buffer =
        CudaBuffer<std::uint32_t>::CopyFromHost(candidate_query_ids);
    CudaBuffer<float> distance_buffer = CudaBuffer<float>::Allocate(num_candidates);

    const std::size_t num_blocks =
        (num_candidates + kWarpsPerBlock - 1) / kWarpsPerBlock;
    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin), "cudaEventRecord");
    rerank_exact_kernel<<<num_blocks, kThreadsPerBlock>>>(
        query_buffer.get(), vector_buffer.get(), query_id_buffer.get(), dim, num_candidates,
        distance_buffer.get());
    ThrowIfCudaError(cudaGetLastError(), "rerank_exact_kernel");
    ThrowIfCudaError(cudaEventRecord(kernel_end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);

    ExactDistanceRunResult result;
    result.query_upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
    result.kernel_ms = static_cast<double>(kernel_ms);
    result.device_distances = std::move(distance_buffer);
    return result;
}

template <typename QueryT, typename VectorT>
ExactDistanceRunResult ComputeExactDistancesFromDeviceVectors(
    const std::vector<QueryT>& queries,
    CudaBuffer<VectorT>&& candidate_vector_buffer,
    const std::vector<std::uint32_t>& candidate_query_ids,
    std::size_t dim) {
    const std::size_t num_candidates = candidate_query_ids.size();
    if (num_candidates == 0) {
        return ExactDistanceRunResult{};
    }

    const auto upload_begin = std::chrono::steady_clock::now();
    CudaBuffer<QueryT> query_buffer = CudaBuffer<QueryT>::CopyFromHost(queries);
    const auto upload_end = std::chrono::steady_clock::now();
    CudaBuffer<std::uint32_t> query_id_buffer =
        CudaBuffer<std::uint32_t>::CopyFromHost(candidate_query_ids);
    CudaBuffer<float> distance_buffer = CudaBuffer<float>::Allocate(num_candidates);

    const std::size_t num_blocks =
        (num_candidates + kWarpsPerBlock - 1) / kWarpsPerBlock;
    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin), "cudaEventRecord");
    rerank_exact_kernel<<<num_blocks, kThreadsPerBlock>>>(
        query_buffer.get(), candidate_vector_buffer.get(), query_id_buffer.get(), dim,
        num_candidates, distance_buffer.get());
    ThrowIfCudaError(cudaGetLastError(), "rerank_exact_kernel");
    ThrowIfCudaError(cudaEventRecord(kernel_end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);

    ExactDistanceRunResult result;
    result.query_upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
    result.kernel_ms = static_cast<double>(kernel_ms);
    result.device_distances = std::move(distance_buffer);
    return result;
}

template <typename QueryT, typename VectorT>
ExactDistanceRunResult ComputeExactDistancesFromDeviceVectors(
    const std::vector<QueryT>& queries,
    CudaBuffer<VectorT>&& candidate_vector_buffer,
    const CudaBuffer<std::uint32_t>& candidate_query_ids,
    std::size_t num_candidates,
    std::size_t dim) {
    if (num_candidates == 0) {
        return ExactDistanceRunResult{};
    }

    const auto upload_begin = std::chrono::steady_clock::now();
    CudaBuffer<QueryT> query_buffer = CudaBuffer<QueryT>::CopyFromHost(queries);
    const auto upload_end = std::chrono::steady_clock::now();
    CudaBuffer<float> distance_buffer = CudaBuffer<float>::Allocate(num_candidates);

    const std::size_t num_blocks =
        (num_candidates + kWarpsPerBlock - 1) / kWarpsPerBlock;
    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin), "cudaEventRecord");
    rerank_exact_kernel<<<num_blocks, kThreadsPerBlock>>>(
        query_buffer.get(), candidate_vector_buffer.get(), candidate_query_ids.get(), dim,
        num_candidates, distance_buffer.get());
    ThrowIfCudaError(cudaGetLastError(), "rerank_exact_kernel");
    ThrowIfCudaError(cudaEventRecord(kernel_end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);

    ExactDistanceRunResult result;
    result.query_upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
    result.kernel_ms = static_cast<double>(kernel_ms);
    result.device_distances = std::move(distance_buffer);
    return result;
}

template <typename VectorT>
CudaBuffer<VectorT> UnpackCandidateVectorsFromDevicePages(
    const DevicePageReadResult& page_result,
    const std::vector<std::size_t>& candidate_page_indices,
    const std::vector<FlatRerankCandidate>& flat_candidates,
    const VectorPageLayout& layout,
    double* out_unpack_ms) {
    const std::size_t num_candidates = flat_candidates.size();
    const std::size_t dim = layout.vector_bytes() / sizeof(VectorT);
    CudaBuffer<VectorT> unpacked = CudaBuffer<VectorT>::Allocate(num_candidates * dim);
    if (num_candidates == 0) {
        if (out_unpack_ms != nullptr) {
            *out_unpack_ms = 0.0;
        }
        return unpacked;
    }

    std::vector<std::uint32_t> page_indices(num_candidates, 0);
    std::vector<std::uint32_t> slot_ids(num_candidates, 0);
    for (std::size_t i = 0; i < num_candidates; ++i) {
        page_indices[i] = static_cast<std::uint32_t>(candidate_page_indices[i]);
        slot_ids[i] = flat_candidates[i].slot_id;
    }

    CudaBuffer<std::uint32_t> page_index_buffer =
        CudaBuffer<std::uint32_t>::CopyFromHost(page_indices);
    CudaBuffer<std::uint32_t> slot_buffer =
        CudaBuffer<std::uint32_t>::CopyFromHost(slot_ids);

    const std::size_t total_elements = num_candidates * dim;
    const std::size_t block_size = 256;
    const std::size_t grid_size =
        std::max<std::size_t>(1, (total_elements + block_size - 1) / block_size);

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(begin), "cudaEventRecord");
    unpack_candidate_vectors_kernel<<<grid_size, block_size>>>(
        page_result.page_bytes.get(), layout.page_size_bytes(), page_index_buffer.get(),
        slot_buffer.get(), layout.vector_bytes(), dim, num_candidates, unpacked.get());
    ThrowIfCudaError(cudaGetLastError(), "unpack_candidate_vectors_kernel");
    ThrowIfCudaError(cudaEventRecord(end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");
    float unpack_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&unpack_ms, begin, end), "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    if (out_unpack_ms != nullptr) {
        *out_unpack_ms = static_cast<double>(unpack_ms);
    }
    return unpacked;
}

template <typename VectorT>
CudaBuffer<VectorT> UnpackCandidateVectorsFromLinearDevicePages(
    const DevicePageReadResult& page_result,
    const CudaBuffer<std::uint32_t>& slot_ids,
    std::size_t num_candidates,
    const VectorPageLayout& layout,
    double* out_unpack_ms) {
    const std::size_t dim = layout.vector_bytes() / sizeof(VectorT);
    CudaBuffer<VectorT> unpacked = CudaBuffer<VectorT>::Allocate(num_candidates * dim);
    if (num_candidates == 0) {
        if (out_unpack_ms != nullptr) {
            *out_unpack_ms = 0.0;
        }
        return unpacked;
    }

    const std::size_t total_elements = num_candidates * dim;
    const std::size_t block_size = 256;
    const std::size_t grid_size =
        std::max<std::size_t>(1, (total_elements + block_size - 1) / block_size);

    cudaEvent_t begin = nullptr;
    cudaEvent_t end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(begin), "cudaEventRecord");
    unpack_candidate_vectors_linear_pages_kernel<<<grid_size, block_size>>>(
        page_result.page_bytes.get(), layout.page_size_bytes(), slot_ids.get(),
        layout.vector_bytes(), dim, num_candidates, unpacked.get());
    ThrowIfCudaError(cudaGetLastError(), "unpack_candidate_vectors_linear_pages_kernel");
    ThrowIfCudaError(cudaEventRecord(end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(end), "cudaEventSynchronize");
    float unpack_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&unpack_ms, begin, end), "cudaEventElapsedTime");
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    if (out_unpack_ms != nullptr) {
        *out_unpack_ms = static_cast<double>(unpack_ms);
    }
    return unpacked;
}

struct UploadedFloatQueries {
    CudaBuffer<float> device_queries;
    double upload_ms = 0.0;
};

UploadedFloatQueries UploadFloatQueries(const std::vector<float>& queries) {
    const auto upload_begin = std::chrono::steady_clock::now();
    UploadedFloatQueries result;
    result.device_queries = CudaBuffer<float>::CopyFromHost(queries);
    const auto upload_end = std::chrono::steady_clock::now();
    result.upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
    return result;
}

template <typename VectorT>
double RunExactDistanceKernelFloatQueries(const CudaBuffer<float>& device_queries,
                                          const CudaBuffer<VectorT>& candidate_vectors,
                                          const CudaBuffer<std::uint32_t>& candidate_query_ids,
                                          std::size_t num_candidates,
                                          std::size_t dim,
                                          CudaBuffer<float>* out_distances) {
    *out_distances = CudaBuffer<float>::Allocate(num_candidates);
    if (num_candidates == 0) {
        return 0.0;
    }

    const std::size_t num_blocks =
        (num_candidates + kWarpsPerBlock - 1) / kWarpsPerBlock;
    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin), "cudaEventRecord");
    if constexpr (std::is_same_v<VectorT, float>) {
        LaunchRerankExactKernelFloat32Specialized(
            device_queries.get(), candidate_vectors.get(), candidate_query_ids.get(), dim,
            num_candidates, out_distances->get(), num_blocks);
    } else {
        rerank_exact_kernel<<<num_blocks, kThreadsPerBlock>>>(
            device_queries.get(), candidate_vectors.get(), candidate_query_ids.get(), dim,
            num_candidates, out_distances->get());
    }
    ThrowIfCudaError(cudaGetLastError(), "rerank_exact_kernel");
    ThrowIfCudaError(cudaEventRecord(kernel_end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);
    return static_cast<double>(kernel_ms);
}

RerankBatchResult FinalizeFusedRerankResults(std::size_t num_queries,
                                             const RerankExactParams& params,
                                             CudaBuffer<float>&& device_distances,
                                             CudaBuffer<std::uint32_t>&& node_ids,
                                             std::size_t requested_pages,
                                             std::size_t exact_distance_count,
                                             std::size_t bound_filtered_count,
                                             double prepare_ms,
                                             double io_ms,
                                             double unpack_ms,
                                             double query_upload_ms,
                                             double exact_kernel_ms,
                                             double pq2_bound_filter_ms,
                                             double pq2_threshold_update_ms,
                                             const std::chrono::steady_clock::time_point& total_begin,
                                             bool already_sorted,
                                             RerankBatchProfile* out_profile) {
    RerankBatchResult result;
    result.queries.resize(num_queries);
    result.stats.io_pages = requested_pages;
    result.stats.exact_distance_count = exact_distance_count;
    result.stats.bound_filtered_count = bound_filtered_count;
    if (num_queries == 0 || params.top_n == 0) {
        return result;
    }

    std::vector<float> sorted_distances;
    std::vector<std::uint32_t> sorted_node_ids;
    double sort_ms = 0.0;
    double result_download_ms = 0.0;
    if (already_sorted) {
        const auto download_begin = std::chrono::steady_clock::now();
        sorted_distances = device_distances.CopyToHost();
        sorted_node_ids = node_ids.CopyToHost();
        const auto download_end = std::chrono::steady_clock::now();
        result_download_ms =
            std::chrono::duration<double, std::milli>(download_end - download_begin).count();
    } else if (params.top_n <= kMaxRerankSortCandidates) {
        std::vector<std::uint32_t> host_query_offsets(num_queries + 1, 0);
        for (std::size_t query_id = 0; query_id <= num_queries; ++query_id) {
            host_query_offsets[query_id] = static_cast<std::uint32_t>(query_id * params.top_n);
        }
        CudaBuffer<std::uint32_t> query_offset_buffer =
            CudaBuffer<std::uint32_t>::CopyFromHost(host_query_offsets);
        const std::size_t padded_count = NextPowerOfTwo(params.top_n);

        cudaEvent_t sort_begin = nullptr;
        cudaEvent_t sort_end = nullptr;
        ThrowIfCudaError(cudaEventCreate(&sort_begin), "cudaEventCreate");
        ThrowIfCudaError(cudaEventCreate(&sort_end), "cudaEventCreate");
        ThrowIfCudaError(cudaEventRecord(sort_begin), "cudaEventRecord");
        rerank_bitonic_sort_kernel<<<num_queries, kRerankSortThreads>>>(
            device_distances.get(), node_ids.get(), query_offset_buffer.get(), padded_count);
        ThrowIfCudaError(cudaGetLastError(), "rerank_bitonic_sort_kernel");
        ThrowIfCudaError(cudaEventRecord(sort_end), "cudaEventRecord");
        ThrowIfCudaError(cudaEventSynchronize(sort_end), "cudaEventSynchronize");
        float sort_kernel_ms = 0.0f;
        ThrowIfCudaError(cudaEventElapsedTime(&sort_kernel_ms, sort_begin, sort_end),
                         "cudaEventElapsedTime");
        cudaEventDestroy(sort_begin);
        cudaEventDestroy(sort_end);
        sort_ms = static_cast<double>(sort_kernel_ms);

        const auto download_begin = std::chrono::steady_clock::now();
        sorted_distances = device_distances.CopyToHost();
        sorted_node_ids = node_ids.CopyToHost();
        const auto download_end = std::chrono::steady_clock::now();
        result_download_ms =
            std::chrono::duration<double, std::milli>(download_end - download_begin).count();
    } else {
        const auto download_begin = std::chrono::steady_clock::now();
        sorted_distances = device_distances.CopyToHost();
        sorted_node_ids = node_ids.CopyToHost();
        const auto download_end = std::chrono::steady_clock::now();
        result_download_ms =
            std::chrono::duration<double, std::milli>(download_end - download_begin).count();

        const auto sort_begin = std::chrono::steady_clock::now();
        auto candidate_less = [](float lhs_distance,
                                 std::uint32_t lhs_node_id,
                                 float rhs_distance,
                                 std::uint32_t rhs_node_id) {
            if (lhs_distance != rhs_distance) {
                return lhs_distance < rhs_distance;
            }
            return lhs_node_id < rhs_node_id;
        };
        for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
            const std::size_t begin = query_id * params.top_n;
            const std::size_t end = begin + params.top_n;
            for (std::size_t i = begin; i < end; ++i) {
                for (std::size_t j = i + 1; j < end; ++j) {
                    if (candidate_less(sorted_distances[j], sorted_node_ids[j],
                                       sorted_distances[i], sorted_node_ids[i])) {
                        std::swap(sorted_distances[i], sorted_distances[j]);
                        std::swap(sorted_node_ids[i], sorted_node_ids[j]);
                    }
                }
            }
        }
        const auto sort_end = std::chrono::steady_clock::now();
        sort_ms =
            std::chrono::duration<double, std::milli>(sort_end - sort_begin).count();
    }

    double sorted_candidates_materialize_ms = 0.0;
    double rerank_topk_extract_ms = 0.0;
    if (already_sorted && params.top_n == params.top_k) {
        // Persistent learned rerank maintains the final top-k on device; host only materializes
        // the output vectors needed by the eval/API result.
        const auto sorted_materialize_begin = std::chrono::steady_clock::now();
        for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
            const std::size_t begin = query_id * params.top_n;
            const std::size_t end = begin + params.top_n;
            auto& sorted = result.queries[query_id].sorted_candidates;
            auto& topk = result.queries[query_id].topk;
            sorted.reserve(params.top_n);
            topk.reserve(params.top_k);
            for (std::size_t flat_index = begin; flat_index < end; ++flat_index) {
                RankedCandidate candidate{
                    sorted_distances[flat_index],
                    sorted_node_ids[flat_index],
                    sorted_node_ids[flat_index] != kInvalidNodeId,
                };
                sorted.push_back(candidate);
                if (candidate.valid()) {
                    topk.push_back(candidate);
                }
            }
        }
        const auto sorted_materialize_end = std::chrono::steady_clock::now();
        sorted_candidates_materialize_ms =
            std::chrono::duration<double, std::milli>(sorted_materialize_end -
                                                      sorted_materialize_begin)
                .count();
    } else {
        for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
            const std::size_t begin = query_id * params.top_n;
            const std::size_t end = begin + params.top_n;
            auto& sorted = result.queries[query_id].sorted_candidates;
            sorted.reserve(params.top_n);
            const auto sorted_materialize_begin = std::chrono::steady_clock::now();
            for (std::size_t flat_index = begin; flat_index < end; ++flat_index) {
                sorted.push_back(RankedCandidate{
                    sorted_distances[flat_index],
                    sorted_node_ids[flat_index],
                    sorted_node_ids[flat_index] != kInvalidNodeId,
                });
            }
            const auto sorted_materialize_end = std::chrono::steady_clock::now();
            const auto topk_extract_begin = std::chrono::steady_clock::now();
            for (const RankedCandidate& candidate : sorted) {
                if (!candidate.valid()) {
                    continue;
                }
                result.queries[query_id].topk.push_back(candidate);
                if (result.queries[query_id].topk.size() == params.top_k) {
                    break;
                }
            }
            const auto topk_extract_end = std::chrono::steady_clock::now();
            sorted_candidates_materialize_ms +=
                std::chrono::duration<double, std::milli>(sorted_materialize_end -
                                                          sorted_materialize_begin)
                    .count();
            rerank_topk_extract_ms +=
                std::chrono::duration<double, std::milli>(topk_extract_end - topk_extract_begin)
                    .count();
        }
    }

    const auto total_end = std::chrono::steady_clock::now();
    result.profile.total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_begin).count();
    result.profile.pq2_bound_filter_ms = pq2_bound_filter_ms;
    result.profile.pq2_threshold_update_ms = pq2_threshold_update_ms;
    result.profile.prepare_ms = prepare_ms;
    result.profile.io_ms = io_ms;
    result.profile.unpack_ms = unpack_ms;
    result.profile.query_upload_ms = query_upload_ms;
    result.profile.exact_distance_kernel_ms = exact_kernel_ms;
    result.profile.exact_distance_total_ms = query_upload_ms + exact_kernel_ms;
    result.profile.sort_ms = sort_ms;
    result.profile.result_download_ms = result_download_ms;
    result.profile.rerank_topk_extract_ms = rerank_topk_extract_ms;
    result.profile.sorted_candidates_materialize_ms = sorted_candidates_materialize_ms;
    result.profile.requested_top_n = params.top_n;
    result.profile.effective_top_n = params.top_n;
    result.profile.exact_candidates = exact_distance_count;
    result.profile.bound_filtered_candidates = bound_filtered_count;
    result.profile.rerank_batches = 1;
    result.profile.rerank_gpu_sort_batches =
        (already_sorted || params.top_n <= kMaxRerankSortCandidates) ? 1 : 0;
    result.profile.rerank_host_sort_batches =
        (already_sorted || params.top_n <= kMaxRerankSortCandidates) ? 0 : 1;
    if (out_profile != nullptr) {
        out_profile->total_ms += result.profile.total_ms;
        out_profile->pq2_bound_filter_ms += result.profile.pq2_bound_filter_ms;
        out_profile->pq2_threshold_update_ms += result.profile.pq2_threshold_update_ms;
        out_profile->prepare_ms += result.profile.prepare_ms;
        out_profile->io_ms += result.profile.io_ms;
        out_profile->unpack_ms += result.profile.unpack_ms;
        out_profile->query_upload_ms += result.profile.query_upload_ms;
        out_profile->exact_distance_kernel_ms += result.profile.exact_distance_kernel_ms;
        out_profile->exact_distance_total_ms += result.profile.exact_distance_total_ms;
        out_profile->sort_ms += result.profile.sort_ms;
        out_profile->result_download_ms += result.profile.result_download_ms;
        out_profile->rerank_topk_extract_ms += result.profile.rerank_topk_extract_ms;
        out_profile->sorted_candidates_materialize_ms +=
            result.profile.sorted_candidates_materialize_ms;
        out_profile->requested_top_n += result.profile.requested_top_n;
        out_profile->effective_top_n += result.profile.effective_top_n;
        out_profile->exact_candidates += result.profile.exact_candidates;
        out_profile->bound_filtered_candidates += result.profile.bound_filtered_candidates;
        out_profile->rerank_batches += result.profile.rerank_batches;
        out_profile->rerank_gpu_sort_batches += result.profile.rerank_gpu_sort_batches;
        out_profile->rerank_host_sort_batches += result.profile.rerank_host_sort_batches;
    }
    return result;
}

template <typename QueryT>
RerankBatchResult RunBatchImpl(const SearchResources& resources,
                               const std::vector<TopologySearchResult>& topology_results,
                               const std::vector<QueryT>& queries,
                               std::size_t num_queries,
                               const RerankExactParams& params,
                               const char* context,
                               RerankBatchProfile* out_profile) {
    const auto total_begin = std::chrono::steady_clock::now();
    if (!resources.has_vector_store()) {
        throw std::runtime_error(
            BuildErrorMessage(context, "Vector store must be loaded before rerank."));
    }
    const auto& header = resources.vector_store_header();
    const auto& layout = resources.vector_store_layout();
    if (topology_results.size() != num_queries) {
        throw std::runtime_error(BuildErrorMessage(
            context, "topology_results.size() must equal num_queries."));
    }
    if (queries.size() != num_queries * header.dim) {
        throw std::runtime_error(BuildErrorMessage(
            context, "query buffer size must equal num_queries * dim."));
    }

    std::vector<FlatRerankCandidate> flat_candidates;
    const auto prepare_begin = std::chrono::steady_clock::now();
    flat_candidates.reserve(num_queries * params.top_n);
    std::vector<std::uint32_t> query_offsets(num_queries + 1, 0);
    std::size_t max_candidates_per_query = 0;
    for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
        query_offsets[query_id] = static_cast<std::uint32_t>(flat_candidates.size());
        const std::size_t limit =
            std::min(params.top_n, topology_results[query_id].sorted_candidates.size());
        for (std::size_t rank = 0; rank < limit; ++rank) {
            const RankedCandidate& candidate = topology_results[query_id].sorted_candidates[rank];
            if (!candidate.valid()) {
                continue;
            }
            const VectorPageAddress address = layout.Resolve(candidate.node_id);
            flat_candidates.push_back(FlatRerankCandidate{
                static_cast<std::uint32_t>(query_id),
                candidate.node_id,
                address.page_id,
                address.slot_id,
            });
        }
        const std::size_t query_candidate_count =
            flat_candidates.size() - static_cast<std::size_t>(query_offsets[query_id]);
        max_candidates_per_query =
            std::max(max_candidates_per_query, query_candidate_count);
    }
    query_offsets[num_queries] = static_cast<std::uint32_t>(flat_candidates.size());

    RerankBatchResult result;
    result.queries.resize(num_queries);
    result.stats.exact_distance_count = flat_candidates.size();
    if (flat_candidates.empty()) {
        if (out_profile != nullptr) {
            out_profile->total_ms += 0.0;
        }
        return result;
    }

    std::vector<std::uint64_t> unique_pages;
    unique_pages.reserve(flat_candidates.size());
    std::unordered_map<std::uint64_t, std::size_t> page_to_index;
    std::vector<std::size_t> candidate_page_indices(flat_candidates.size(), 0);
    for (std::size_t i = 0; i < flat_candidates.size(); ++i) {
        const auto [iter, inserted] =
            page_to_index.emplace(flat_candidates[i].page_id, unique_pages.size());
        if (inserted) {
            unique_pages.push_back(flat_candidates[i].page_id);
        }
        candidate_page_indices[i] = iter->second;
    }
    result.stats.io_pages = unique_pages.size();
    const auto prepare_end = std::chrono::steady_clock::now();

    std::vector<std::uint32_t> candidate_query_ids(flat_candidates.size(), 0);
    for (std::size_t i = 0; i < flat_candidates.size(); ++i) {
        candidate_query_ids[i] = flat_candidates[i].query_id;
    }

    const VectorPageProvider& page_provider = resources.vector_page_provider();
    ExactDistanceRunResult exact_run;
    double io_ms = 0.0;
    double unpack_ms = 0.0;
    if (page_provider.SupportsDeviceReads()) {
        const DevicePageReadResult device_pages = page_provider.ReadPagesToDevice(
            resources.vector_store_path(), unique_pages, layout.header_bytes(),
            layout.page_size_bytes());
        io_ms = device_pages.io_ms;
        switch (static_cast<ScalarKind>(header.scalar_kind)) {
            case ScalarKind::kFloat32: {
                CudaBuffer<float> candidate_vectors =
                    UnpackCandidateVectorsFromDevicePages<float>(
                        device_pages, candidate_page_indices, flat_candidates, layout,
                        &unpack_ms);
                exact_run = ComputeExactDistancesFromDeviceVectors<QueryT, float>(
                    queries, std::move(candidate_vectors), candidate_query_ids, header.dim);
                break;
            }
            case ScalarKind::kUint8: {
                CudaBuffer<std::uint8_t> candidate_vectors =
                    UnpackCandidateVectorsFromDevicePages<std::uint8_t>(
                        device_pages, candidate_page_indices, flat_candidates, layout,
                        &unpack_ms);
                exact_run = ComputeExactDistancesFromDeviceVectors<QueryT, std::uint8_t>(
                    queries, std::move(candidate_vectors), candidate_query_ids, header.dim);
                break;
            }
            case ScalarKind::kInt8: {
                CudaBuffer<std::int8_t> candidate_vectors =
                    UnpackCandidateVectorsFromDevicePages<std::int8_t>(
                        device_pages, candidate_page_indices, flat_candidates, layout,
                        &unpack_ms);
                exact_run = ComputeExactDistancesFromDeviceVectors<QueryT, std::int8_t>(
                    queries, std::move(candidate_vectors), candidate_query_ids, header.dim);
                break;
            }
            default:
                throw std::runtime_error(
                    BuildErrorMessage(context, "Unsupported vector store scalar kind."));
        }
    } else {
        const auto io_begin = std::chrono::steady_clock::now();
        const std::vector<std::uint8_t> page_bytes = page_provider.ReadPages(
            resources.vector_store_path(), unique_pages, layout.header_bytes(),
            layout.page_size_bytes());
        const auto io_end = std::chrono::steady_clock::now();

        const auto unpack_begin = std::chrono::steady_clock::now();
        std::vector<std::uint8_t> candidate_vector_bytes(
            flat_candidates.size() * layout.vector_bytes(), 0);
        for (std::size_t i = 0; i < flat_candidates.size(); ++i) {
            const std::size_t page_index = candidate_page_indices[i];
            const std::size_t source_offset =
                page_index * layout.page_size_bytes() +
                static_cast<std::size_t>(flat_candidates[i].slot_id) * layout.vector_bytes();
            std::memcpy(candidate_vector_bytes.data() + i * layout.vector_bytes(),
                        page_bytes.data() + source_offset, layout.vector_bytes());
        }
        const auto unpack_end = std::chrono::steady_clock::now();
        io_ms = std::chrono::duration<double, std::milli>(io_end - io_begin).count();
        unpack_ms =
            std::chrono::duration<double, std::milli>(unpack_end - unpack_begin).count();

        switch (static_cast<ScalarKind>(header.scalar_kind)) {
            case ScalarKind::kFloat32:
                exact_run = ComputeExactDistances<QueryT, float>(
                    queries, candidate_vector_bytes, candidate_query_ids, header.dim);
                break;
            case ScalarKind::kUint8:
                exact_run = ComputeExactDistances<QueryT, std::uint8_t>(
                    queries, candidate_vector_bytes, candidate_query_ids, header.dim);
                break;
            case ScalarKind::kInt8:
                exact_run = ComputeExactDistances<QueryT, std::int8_t>(
                    queries, candidate_vector_bytes, candidate_query_ids, header.dim);
                break;
            default:
                throw std::runtime_error(
                    BuildErrorMessage(context, "Unsupported vector store scalar kind."));
        }
    }

    std::vector<float> sorted_distances;
    std::vector<std::uint32_t> sorted_node_ids;
    double sort_ms = 0.0;
    double result_download_ms = 0.0;
    if (max_candidates_per_query <= kMaxRerankSortCandidates) {
        std::vector<std::uint32_t> flat_node_ids(flat_candidates.size(), kInvalidNodeId);
        for (std::size_t i = 0; i < flat_candidates.size(); ++i) {
            flat_node_ids[i] = flat_candidates[i].node_id;
        }
        CudaBuffer<std::uint32_t> node_id_buffer =
            CudaBuffer<std::uint32_t>::CopyFromHost(flat_node_ids);
        CudaBuffer<std::uint32_t> query_offset_buffer =
            CudaBuffer<std::uint32_t>::CopyFromHost(query_offsets);
        const std::size_t padded_count = NextPowerOfTwo(max_candidates_per_query);

        cudaEvent_t sort_begin = nullptr;
        cudaEvent_t sort_end = nullptr;
        ThrowIfCudaError(cudaEventCreate(&sort_begin), "cudaEventCreate");
        ThrowIfCudaError(cudaEventCreate(&sort_end), "cudaEventCreate");
        ThrowIfCudaError(cudaEventRecord(sort_begin), "cudaEventRecord");
        rerank_bitonic_sort_kernel<<<num_queries, kRerankSortThreads>>>(
            exact_run.device_distances.get(), node_id_buffer.get(), query_offset_buffer.get(),
            padded_count);
        ThrowIfCudaError(cudaGetLastError(), "rerank_bitonic_sort_kernel");
        ThrowIfCudaError(cudaEventRecord(sort_end), "cudaEventRecord");
        ThrowIfCudaError(cudaEventSynchronize(sort_end), "cudaEventSynchronize");
        float sort_kernel_ms = 0.0f;
        ThrowIfCudaError(cudaEventElapsedTime(&sort_kernel_ms, sort_begin, sort_end),
                         "cudaEventElapsedTime");
        cudaEventDestroy(sort_begin);
        cudaEventDestroy(sort_end);
        sort_ms = static_cast<double>(sort_kernel_ms);

        const auto download_begin = std::chrono::steady_clock::now();
        sorted_distances = exact_run.device_distances.CopyToHost();
        sorted_node_ids = node_id_buffer.CopyToHost();
        const auto download_end = std::chrono::steady_clock::now();
        result_download_ms =
            std::chrono::duration<double, std::milli>(download_end - download_begin).count();
    } else {
        const auto download_begin = std::chrono::steady_clock::now();
        sorted_distances = exact_run.device_distances.CopyToHost();
        const auto download_end = std::chrono::steady_clock::now();
        result_download_ms =
            std::chrono::duration<double, std::milli>(download_end - download_begin).count();
        sorted_node_ids.resize(flat_candidates.size(), kInvalidNodeId);
        for (std::size_t i = 0; i < flat_candidates.size(); ++i) {
            sorted_node_ids[i] = flat_candidates[i].node_id;
        }

        const auto sort_begin = std::chrono::steady_clock::now();
        auto candidate_less = [](float lhs_distance,
                                 std::uint32_t lhs_node_id,
                                 float rhs_distance,
                                 std::uint32_t rhs_node_id) {
            if (lhs_distance != rhs_distance) {
                return lhs_distance < rhs_distance;
            }
            return lhs_node_id < rhs_node_id;
        };
        for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
            const std::size_t begin = query_offsets[query_id];
            const std::size_t end = query_offsets[query_id + 1];
            for (std::size_t i = begin; i < end; ++i) {
                for (std::size_t j = i + 1; j < end; ++j) {
                    if (candidate_less(sorted_distances[j], sorted_node_ids[j],
                                       sorted_distances[i], sorted_node_ids[i])) {
                        std::swap(sorted_distances[i], sorted_distances[j]);
                        std::swap(sorted_node_ids[i], sorted_node_ids[j]);
                    }
                }
            }
        }
        const auto sort_end = std::chrono::steady_clock::now();
        sort_ms =
            std::chrono::duration<double, std::milli>(sort_end - sort_begin).count();
    }

    double sorted_candidates_materialize_ms = 0.0;
    double rerank_topk_extract_ms = 0.0;
    for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
        const std::size_t begin = query_offsets[query_id];
        const std::size_t end = query_offsets[query_id + 1];
        auto& sorted = result.queries[query_id].sorted_candidates;
        sorted.reserve(end - begin);
        const auto sorted_materialize_begin = std::chrono::steady_clock::now();
        for (std::size_t flat_index = begin; flat_index < end; ++flat_index) {
            sorted.push_back(RankedCandidate{
                sorted_distances[flat_index],
                sorted_node_ids[flat_index],
                sorted_node_ids[flat_index] != kInvalidNodeId,
            });
        }
        const auto sorted_materialize_end = std::chrono::steady_clock::now();
        const auto topk_extract_begin = std::chrono::steady_clock::now();
        for (const RankedCandidate& candidate : sorted) {
            if (!candidate.valid()) {
                continue;
            }
            result.queries[query_id].topk.push_back(candidate);
            if (result.queries[query_id].topk.size() == params.top_k) {
                break;
            }
        }
        const auto topk_extract_end = std::chrono::steady_clock::now();
        sorted_candidates_materialize_ms +=
            std::chrono::duration<double, std::milli>(sorted_materialize_end -
                                                      sorted_materialize_begin)
                .count();
        rerank_topk_extract_ms +=
            std::chrono::duration<double, std::milli>(topk_extract_end - topk_extract_begin)
                .count();
    }
    const auto total_end = std::chrono::steady_clock::now();

    result.profile.total_ms =
        std::chrono::duration<double, std::milli>(total_end - total_begin).count();
    result.profile.prepare_ms =
        std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count();
    result.profile.io_ms = io_ms;
    result.profile.unpack_ms = unpack_ms;
    result.profile.query_upload_ms = exact_run.query_upload_ms;
    result.profile.exact_distance_kernel_ms = exact_run.kernel_ms;
    result.profile.exact_distance_total_ms =
        exact_run.query_upload_ms + exact_run.kernel_ms;
    result.profile.sort_ms = sort_ms;
    result.profile.result_download_ms = result_download_ms;
    result.profile.rerank_topk_extract_ms = rerank_topk_extract_ms;
    result.profile.sorted_candidates_materialize_ms = sorted_candidates_materialize_ms;
    if (out_profile != nullptr) {
        out_profile->total_ms += result.profile.total_ms;
        out_profile->prepare_ms += result.profile.prepare_ms;
        out_profile->io_ms += result.profile.io_ms;
        out_profile->unpack_ms += result.profile.unpack_ms;
        out_profile->query_upload_ms += result.profile.query_upload_ms;
        out_profile->exact_distance_kernel_ms += result.profile.exact_distance_kernel_ms;
        out_profile->exact_distance_total_ms += result.profile.exact_distance_total_ms;
        out_profile->sort_ms += result.profile.sort_ms;
        out_profile->result_download_ms += result.profile.result_download_ms;
        out_profile->rerank_topk_extract_ms += result.profile.rerank_topk_extract_ms;
        out_profile->sorted_candidates_materialize_ms +=
            result.profile.sorted_candidates_materialize_ms;
    }

    return result;
}

}  // namespace

namespace detail {

RerankBatchResult RunBatchFloat32FromDeviceTopologyLinear(
    const SearchResources& resources,
    const DeviceTopologyBatchResult& topology_result,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const RerankExactParams& params,
    RerankBatchProfile* out_profile) {
    const auto total_begin = std::chrono::steady_clock::now();
    const auto& header = resources.vector_store_header();
    const auto& layout = resources.vector_store_layout();
    const std::size_t flat_count = num_queries * params.top_n;
    if (flat_count == 0) {
        return RerankBatchResult{};
    }
    const UploadedFloatQueries uploaded_queries = UploadFloatQueries(queries);
    const auto prepare_begin = std::chrono::steady_clock::now();
    CudaBuffer<std::uint64_t> page_ids = CudaBuffer<std::uint64_t>::Allocate(flat_count);
    CudaBuffer<std::uint32_t> slot_ids = CudaBuffer<std::uint32_t>::Allocate(flat_count);
    CudaBuffer<std::uint32_t> node_ids = CudaBuffer<std::uint32_t>::Allocate(flat_count);
    CudaBuffer<std::uint32_t> query_ids = CudaBuffer<std::uint32_t>::Allocate(flat_count);
    CudaBuffer<std::uint32_t> valid_count_buffer = CudaBuffer<std::uint32_t>::Allocate(1);
    ThrowIfCudaError(cudaMemset(valid_count_buffer.get(), 0, sizeof(std::uint32_t)), "cudaMemset");

    prepare_topology_candidates_for_fused_rerank_kernel<<<num_queries, 128>>>(
        topology_result.candidate_buffer.get(), topology_result.candidate_capacity, params.top_n,
        layout.vectors_per_page(), page_ids.get(), slot_ids.get(), node_ids.get(),
        query_ids.get(), valid_count_buffer.get());
    ThrowIfCudaError(cudaGetLastError(), "prepare_topology_candidates_for_fused_rerank_kernel");
    ThrowIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    const auto prepare_end = std::chrono::steady_clock::now();

    std::uint32_t valid_count = 0;
    ThrowIfCudaError(cudaMemcpy(&valid_count, valid_count_buffer.get(), sizeof(std::uint32_t),
                                cudaMemcpyDeviceToHost),
                     "cudaMemcpyDeviceToHost");

    double unpack_ms = 0.0;
    double exact_kernel_ms = 0.0;
    double io_ms = 0.0;
    CudaBuffer<float> distance_buffer;
    const VectorPageProvider& page_provider = resources.vector_page_provider();
    const bool use_bam_direct = page_provider.DeviceReadHandle() != nullptr;
    if (use_bam_direct) {
        io_ms = RunBamFusedExactDistanceFloatQueries(
            page_provider, static_cast<ScalarKind>(header.scalar_kind),
            uploaded_queries.device_queries, page_ids, slot_ids, node_ids, query_ids, flat_count,
            layout, header.dim, &distance_buffer);
    } else {
        switch (static_cast<ScalarKind>(header.scalar_kind)) {
            case ScalarKind::kFloat32: {
                CudaBuffer<float> candidate_vectors;
                const DevicePageReadResult device_pages = page_provider.ReadPagesToDevice(
                    resources.vector_store_path(), page_ids, flat_count, layout.header_bytes(),
                    layout.page_size_bytes());
                io_ms = device_pages.io_ms;
                candidate_vectors = UnpackCandidateVectorsFromLinearDevicePages<float>(
                    device_pages, slot_ids, flat_count, layout, &unpack_ms);
                exact_kernel_ms = RunExactDistanceKernelFloatQueries(
                    uploaded_queries.device_queries, candidate_vectors, query_ids, flat_count,
                    header.dim, &distance_buffer);
                break;
            }
            case ScalarKind::kUint8: {
                CudaBuffer<std::uint8_t> candidate_vectors;
                const DevicePageReadResult device_pages = page_provider.ReadPagesToDevice(
                    resources.vector_store_path(), page_ids, flat_count, layout.header_bytes(),
                    layout.page_size_bytes());
                io_ms = device_pages.io_ms;
                candidate_vectors = UnpackCandidateVectorsFromLinearDevicePages<std::uint8_t>(
                    device_pages, slot_ids, flat_count, layout, &unpack_ms);
                exact_kernel_ms = RunExactDistanceKernelFloatQueries(
                    uploaded_queries.device_queries, candidate_vectors, query_ids, flat_count,
                    header.dim, &distance_buffer);
                break;
            }
            case ScalarKind::kInt8: {
                CudaBuffer<std::int8_t> candidate_vectors;
                const DevicePageReadResult device_pages = page_provider.ReadPagesToDevice(
                    resources.vector_store_path(), page_ids, flat_count, layout.header_bytes(),
                    layout.page_size_bytes());
                io_ms = device_pages.io_ms;
                candidate_vectors = UnpackCandidateVectorsFromLinearDevicePages<std::int8_t>(
                    device_pages, slot_ids, flat_count, layout, &unpack_ms);
                exact_kernel_ms = RunExactDistanceKernelFloatQueries(
                    uploaded_queries.device_queries, candidate_vectors, query_ids, flat_count,
                    header.dim, &distance_buffer);
                break;
            }
            default:
                throw std::runtime_error(BuildErrorMessage("RunBatchFloat32FromDeviceTopology",
                                                           "Unsupported vector store scalar kind."));
        }
    }

    const std::size_t mask_blocks = std::max<std::size_t>(1, (flat_count + 255) / 256);
    mask_invalid_rerank_distances_kernel<<<mask_blocks, 256>>>(
        node_ids.get(), flat_count, distance_buffer.get());
    ThrowIfCudaError(cudaGetLastError(), "mask_invalid_rerank_distances_kernel");
    ThrowIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    return FinalizeFusedRerankResults(
        num_queries, params, std::move(distance_buffer), std::move(node_ids), flat_count,
        valid_count, 0,
        std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count(),
        io_ms, unpack_ms, uploaded_queries.upload_ms, exact_kernel_ms, 0.0, 0.0,
        total_begin, false, out_profile);
}

RerankBatchResult RunBatchFloat32FromDeviceTopologyRankTiled(
    const SearchResources& resources,
    const DeviceTopologyBatchResult& topology_result,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const RerankExactParams& params,
    std::size_t rank_tile_size,
    RerankBatchProfile* out_profile) {
    const auto total_begin = std::chrono::steady_clock::now();
    const auto& header = resources.vector_store_header();
    const auto& layout = resources.vector_store_layout();
    const std::size_t flat_count = num_queries * params.top_n;
    if (flat_count == 0) {
        return RerankBatchResult{};
    }

    const VectorPageProvider& page_provider = resources.vector_page_provider();
    const bool use_bam_direct = page_provider.DeviceReadHandle() != nullptr;
    const UploadedFloatQueries uploaded_queries = UploadFloatQueries(queries);
    const std::size_t effective_rank_tile =
        std::max<std::size_t>(1, std::min(rank_tile_size, params.top_n));
    const std::size_t max_tile_count = num_queries * effective_rank_tile;
    const bool use_bound_filter = params.use_pq2_bound_filter;

    CudaBuffer<float> final_distances = CudaBuffer<float>::Allocate(flat_count);
    CudaBuffer<std::uint32_t> final_node_ids = CudaBuffer<std::uint32_t>::Allocate(flat_count);
    CudaBuffer<std::uint64_t> page_ids = CudaBuffer<std::uint64_t>::Allocate(max_tile_count);
    CudaBuffer<std::uint32_t> slot_ids = CudaBuffer<std::uint32_t>::Allocate(max_tile_count);
    CudaBuffer<std::uint32_t> node_ids = CudaBuffer<std::uint32_t>::Allocate(max_tile_count);
    CudaBuffer<std::uint32_t> query_ids = CudaBuffer<std::uint32_t>::Allocate(max_tile_count);
    CudaBuffer<std::uint32_t> local_ranks;
    CudaBuffer<std::uint32_t> valid_count_buffer = CudaBuffer<std::uint32_t>::Allocate(1);
    CudaBuffer<std::uint32_t> filtered_count_buffer;
    CudaBuffer<float> query_thresholds;
    CudaBuffer<float> query_norm_squares;
    if (use_bound_filter) {
        local_ranks = CudaBuffer<std::uint32_t>::Allocate(max_tile_count);
        filtered_count_buffer = CudaBuffer<std::uint32_t>::Allocate(1);
        std::vector<float> host_query_thresholds(num_queries, std::numeric_limits<float>::infinity());
        std::vector<float> host_query_norm_squares(num_queries, 0.0f);
        const std::size_t dim = header.dim;
        for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
            const float* query_ptr = queries.data() + query_id * dim;
            double sum = 0.0;
            for (std::size_t d = 0; d < dim; ++d) {
                const double value = static_cast<double>(query_ptr[d]);
                sum += value * value;
            }
            host_query_norm_squares[query_id] = static_cast<float>(sum);
        }
        query_thresholds = CudaBuffer<float>::CopyFromHost(host_query_thresholds);
        query_norm_squares = CudaBuffer<float>::CopyFromHost(host_query_norm_squares);
        const std::size_t init_blocks = std::max<std::size_t>(1, (flat_count + 255) / 256);
        fill_buffer_kernel<<<init_blocks, 256>>>(
            final_distances.get(), flat_count, std::numeric_limits<float>::infinity());
        ThrowIfCudaError(cudaGetLastError(), "fill_buffer_kernel<float>");
        fill_buffer_kernel<<<init_blocks, 256>>>(final_node_ids.get(), flat_count, kInvalidNodeId);
        ThrowIfCudaError(cudaGetLastError(), "fill_buffer_kernel<uint32_t>");
        ThrowIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    double prepare_ms = 0.0;
    double io_ms = 0.0;
    double unpack_ms = 0.0;
    double exact_kernel_ms = 0.0;
    double bound_filter_ms = 0.0;
    double threshold_update_ms = 0.0;
    std::size_t total_valid_count = 0;
    std::size_t total_bound_filtered_count = 0;
    std::size_t min_tile_valid_count = 0;
    std::size_t max_tile_valid_count = 0;
    std::size_t min_tile_filtered_count = 0;
    std::size_t max_tile_filtered_count = 0;
    std::size_t nonzero_tiles = 0;
    std::size_t zero_tiles = 0;
    bool has_tile_valid = false;
    bool has_tile_filtered = false;

    for (std::size_t rank_begin = 0; rank_begin < params.top_n; rank_begin += effective_rank_tile) {
        const std::size_t rank_count =
            std::min(effective_rank_tile, params.top_n - rank_begin);
        const std::size_t tile_count = num_queries * rank_count;
        const auto prepare_begin = std::chrono::steady_clock::now();
        ThrowIfCudaError(cudaMemset(valid_count_buffer.get(), 0, sizeof(std::uint32_t)),
                         "cudaMemset");
        if (use_bound_filter) {
            ThrowIfCudaError(cudaMemset(filtered_count_buffer.get(), 0, sizeof(std::uint32_t)),
                             "cudaMemset");
            prepare_topology_candidates_for_rank_tile_compact_with_bound_kernel<<<num_queries, 128>>>(
                topology_result.candidate_buffer.get(), topology_result.candidate_capacity,
                rank_begin, rank_count, layout.vectors_per_page(),
                resources.pq2_error_bounds_fp32().get(), query_norm_squares.get(),
                query_thresholds.get(), page_ids.get(), slot_ids.get(), node_ids.get(),
                query_ids.get(), local_ranks.get(), valid_count_buffer.get(),
                filtered_count_buffer.get());
            ThrowIfCudaError(
                cudaGetLastError(),
                "prepare_topology_candidates_for_rank_tile_compact_with_bound_kernel");
        } else {
            prepare_topology_candidates_for_rank_tile_kernel<<<num_queries, 128>>>(
                topology_result.candidate_buffer.get(), topology_result.candidate_capacity,
                rank_begin, rank_count, layout.vectors_per_page(), page_ids.get(), slot_ids.get(),
                node_ids.get(), query_ids.get(), valid_count_buffer.get());
            ThrowIfCudaError(cudaGetLastError(), "prepare_topology_candidates_for_rank_tile_kernel");
        }
        ThrowIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        const auto prepare_end = std::chrono::steady_clock::now();
        const double tile_prepare_ms =
            std::chrono::duration<double, std::milli>(prepare_end - prepare_begin).count();
        prepare_ms += tile_prepare_ms;
        if (use_bound_filter) {
            bound_filter_ms += tile_prepare_ms;
        }

        std::uint32_t tile_valid_count = 0;
        ThrowIfCudaError(cudaMemcpy(&tile_valid_count, valid_count_buffer.get(),
                                    sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                         "cudaMemcpyDeviceToHost");
        total_valid_count += tile_valid_count;
        if (!has_tile_valid) {
            min_tile_valid_count = tile_valid_count;
            max_tile_valid_count = tile_valid_count;
            has_tile_valid = true;
        } else {
            min_tile_valid_count = std::min<std::size_t>(min_tile_valid_count, tile_valid_count);
            max_tile_valid_count = std::max<std::size_t>(max_tile_valid_count, tile_valid_count);
        }
        if (use_bound_filter) {
            std::uint32_t tile_filtered_count = 0;
            ThrowIfCudaError(cudaMemcpy(&tile_filtered_count, filtered_count_buffer.get(),
                                        sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
                             "cudaMemcpyDeviceToHost");
            total_bound_filtered_count += tile_filtered_count;
            if (!has_tile_filtered) {
                min_tile_filtered_count = tile_filtered_count;
                max_tile_filtered_count = tile_filtered_count;
                has_tile_filtered = true;
            } else {
                min_tile_filtered_count =
                    std::min<std::size_t>(min_tile_filtered_count, tile_filtered_count);
                max_tile_filtered_count =
                    std::max<std::size_t>(max_tile_filtered_count, tile_filtered_count);
            }
        }

        if (tile_valid_count == 0) {
            ++zero_tiles;
            continue;
        }
        ++nonzero_tiles;

        CudaBuffer<float> tile_distances;
        if (use_bam_direct) {
            const double tile_io_ms = RunBamFusedExactDistanceFloatQueries(
                page_provider, static_cast<ScalarKind>(header.scalar_kind),
                uploaded_queries.device_queries, page_ids, slot_ids, node_ids, query_ids,
                tile_valid_count, layout, header.dim, &tile_distances);
            io_ms += tile_io_ms;
        } else {
            switch (static_cast<ScalarKind>(header.scalar_kind)) {
                case ScalarKind::kFloat32: {
                    double tile_io_ms = 0.0;
                    double tile_unpack_ms = 0.0;
                    CudaBuffer<float> candidate_vectors;
                    const DevicePageReadResult device_pages = page_provider.ReadPagesToDevice(
                        resources.vector_store_path(), page_ids, tile_valid_count,
                        layout.header_bytes(), layout.page_size_bytes());
                    tile_io_ms = device_pages.io_ms;
                    candidate_vectors = UnpackCandidateVectorsFromLinearDevicePages<float>(
                        device_pages, slot_ids, tile_valid_count, layout, &tile_unpack_ms);
                    io_ms += tile_io_ms;
                    unpack_ms += tile_unpack_ms;
                    exact_kernel_ms += RunExactDistanceKernelFloatQueries(
                        uploaded_queries.device_queries, candidate_vectors, query_ids,
                        tile_valid_count, header.dim, &tile_distances);
                    break;
                }
                case ScalarKind::kUint8: {
                    double tile_io_ms = 0.0;
                    double tile_unpack_ms = 0.0;
                    CudaBuffer<std::uint8_t> candidate_vectors;
                    const DevicePageReadResult device_pages = page_provider.ReadPagesToDevice(
                        resources.vector_store_path(), page_ids, tile_valid_count,
                        layout.header_bytes(), layout.page_size_bytes());
                    tile_io_ms = device_pages.io_ms;
                    candidate_vectors = UnpackCandidateVectorsFromLinearDevicePages<std::uint8_t>(
                        device_pages, slot_ids, tile_valid_count, layout, &tile_unpack_ms);
                    io_ms += tile_io_ms;
                    unpack_ms += tile_unpack_ms;
                    exact_kernel_ms += RunExactDistanceKernelFloatQueries(
                        uploaded_queries.device_queries, candidate_vectors, query_ids,
                        tile_valid_count, header.dim, &tile_distances);
                    break;
                }
                case ScalarKind::kInt8: {
                    double tile_io_ms = 0.0;
                    double tile_unpack_ms = 0.0;
                    CudaBuffer<std::int8_t> candidate_vectors;
                    const DevicePageReadResult device_pages = page_provider.ReadPagesToDevice(
                        resources.vector_store_path(), page_ids, tile_valid_count,
                        layout.header_bytes(), layout.page_size_bytes());
                    tile_io_ms = device_pages.io_ms;
                    candidate_vectors = UnpackCandidateVectorsFromLinearDevicePages<std::int8_t>(
                        device_pages, slot_ids, tile_valid_count, layout, &tile_unpack_ms);
                    io_ms += tile_io_ms;
                    unpack_ms += tile_unpack_ms;
                    exact_kernel_ms += RunExactDistanceKernelFloatQueries(
                        uploaded_queries.device_queries, candidate_vectors, query_ids,
                        tile_valid_count, header.dim, &tile_distances);
                    break;
                }
                default:
                    throw std::runtime_error(BuildErrorMessage(
                        "RunBatchFloat32FromDeviceTopology",
                        "Unsupported vector store scalar kind."));
            }
        }

        const std::size_t mask_blocks = std::max<std::size_t>(1, (tile_valid_count + 255) / 256);
        mask_invalid_rerank_distances_kernel<<<mask_blocks, 256>>>(
            node_ids.get(), tile_valid_count, tile_distances.get());
        ThrowIfCudaError(cudaGetLastError(), "mask_invalid_rerank_distances_kernel");

        if (use_bound_filter) {
            const std::size_t scatter_blocks =
                std::max<std::size_t>(1, (tile_valid_count + 255) / 256);
            scatter_rank_tile_compact_results_kernel<<<scatter_blocks, 256>>>(
                tile_distances.get(), node_ids.get(), query_ids.get(), local_ranks.get(),
                params.top_n, rank_begin, tile_valid_count, final_distances.get(),
                final_node_ids.get());
            ThrowIfCudaError(cudaGetLastError(), "scatter_rank_tile_compact_results_kernel");
        } else {
            const std::size_t scatter_blocks = std::max<std::size_t>(1, (tile_count + 255) / 256);
            scatter_rank_tile_results_kernel<<<scatter_blocks, 256>>>(
                tile_distances.get(), node_ids.get(), params.top_n, rank_begin, rank_count,
                tile_count, final_distances.get(), final_node_ids.get());
            ThrowIfCudaError(cudaGetLastError(), "scatter_rank_tile_results_kernel");
        }

        if (use_bound_filter) {
            const std::size_t prefix_count =
                std::min(params.top_n, rank_begin + rank_count);
            cudaEvent_t threshold_begin = nullptr;
            cudaEvent_t threshold_end = nullptr;
            ThrowIfCudaError(cudaEventCreate(&threshold_begin), "cudaEventCreate");
            ThrowIfCudaError(cudaEventCreate(&threshold_end), "cudaEventCreate");
            ThrowIfCudaError(cudaEventRecord(threshold_begin), "cudaEventRecord");
            update_query_thresholds_from_final_distances_kernel<<<num_queries, 32>>>(
                final_distances.get(), params.top_n, prefix_count, params.top_k,
                query_thresholds.get());
            ThrowIfCudaError(cudaGetLastError(),
                             "update_query_thresholds_from_final_distances_kernel");
            ThrowIfCudaError(cudaEventRecord(threshold_end), "cudaEventRecord");
            ThrowIfCudaError(cudaEventSynchronize(threshold_end), "cudaEventSynchronize");
            float tile_threshold_ms = 0.0f;
            ThrowIfCudaError(cudaEventElapsedTime(&tile_threshold_ms, threshold_begin,
                                                  threshold_end),
                             "cudaEventElapsedTime");
            cudaEventDestroy(threshold_begin);
            cudaEventDestroy(threshold_end);
            threshold_update_ms += static_cast<double>(tile_threshold_ms);
        } else {
            ThrowIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
        }
    }

    RerankBatchResult result = FinalizeFusedRerankResults(
        num_queries, params, std::move(final_distances), std::move(final_node_ids),
        total_valid_count,
        total_valid_count, total_bound_filtered_count, prepare_ms, io_ms, unpack_ms,
        uploaded_queries.upload_ms, exact_kernel_ms, bound_filter_ms, threshold_update_ms,
        total_begin, false, out_profile);
    result.profile.requested_top_n = params.top_n;
    result.profile.effective_top_n =
        std::min(params.top_n, topology_result.candidate_capacity);
    result.profile.rerank_tile_size = effective_rank_tile;
    result.profile.rerank_bam_direct_batches = use_bam_direct ? 1 : 0;
    result.profile.rerank_prepare_valid_candidates = total_valid_count;
    result.profile.rerank_prepare_valid_candidates_min =
        has_tile_valid ? min_tile_valid_count : 0;
    result.profile.rerank_prepare_valid_candidates_max =
        has_tile_valid ? max_tile_valid_count : 0;
    result.profile.rerank_prepare_filtered_candidates = total_bound_filtered_count;
    result.profile.rerank_prepare_filtered_candidates_min =
        has_tile_filtered ? min_tile_filtered_count : 0;
    result.profile.rerank_prepare_filtered_candidates_max =
        has_tile_filtered ? max_tile_filtered_count : 0;
    result.profile.rerank_prepare_nonzero_tiles = nonzero_tiles;
    result.profile.rerank_prepare_zero_tiles = zero_tiles;
    if (out_profile != nullptr) {
        out_profile->effective_top_n +=
            result.profile.effective_top_n - result.profile.requested_top_n;
        out_profile->rerank_tile_size += result.profile.rerank_tile_size;
        out_profile->rerank_bam_direct_batches += result.profile.rerank_bam_direct_batches;
        out_profile->rerank_prepare_valid_candidates +=
            result.profile.rerank_prepare_valid_candidates;
        out_profile->rerank_prepare_filtered_candidates +=
            result.profile.rerank_prepare_filtered_candidates;
        const std::size_t previous_tile_count =
            out_profile->rerank_prepare_nonzero_tiles + out_profile->rerank_prepare_zero_tiles;
        out_profile->rerank_prepare_nonzero_tiles += result.profile.rerank_prepare_nonzero_tiles;
        out_profile->rerank_prepare_zero_tiles += result.profile.rerank_prepare_zero_tiles;
        bool has_dst_valid = previous_tile_count != 0;
        MergeMinMax(result.profile.rerank_prepare_valid_candidates_min,
                    result.profile.rerank_prepare_valid_candidates_max, has_tile_valid,
                    &out_profile->rerank_prepare_valid_candidates_min,
                    &out_profile->rerank_prepare_valid_candidates_max, &has_dst_valid);
        bool has_dst_filtered = previous_tile_count != 0;
        MergeMinMax(result.profile.rerank_prepare_filtered_candidates_min,
                    result.profile.rerank_prepare_filtered_candidates_max, has_tile_filtered,
                    &out_profile->rerank_prepare_filtered_candidates_min,
                    &out_profile->rerank_prepare_filtered_candidates_max, &has_dst_filtered);
    }
    return result;
}

RerankBatchResult RunBatchFloat32FromDeviceTopologyPersistent(
    const SearchResources& resources,
    const DeviceTopologyBatchResult& topology_result,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const RerankExactParams& params,
    RerankBatchProfile* out_profile) {
    const auto total_begin = std::chrono::steady_clock::now();
    const auto& header = resources.vector_store_header();
    if (static_cast<ScalarKind>(header.scalar_kind) != ScalarKind::kFloat32) {
        throw std::runtime_error(BuildErrorMessage(
            "RunBatchFloat32FromDeviceTopologyPersistent",
            "Persistent rerank currently requires float32 vectors."));
    }
    if (params.rank_tile_size == 0 || params.rank_tile_size > kMaxPersistentTileSize) {
        throw std::runtime_error(BuildErrorMessage(
            "RunBatchFloat32FromDeviceTopologyPersistent",
            "Persistent rerank requires 1 <= rank_tile_size <= 128."));
    }
    if (params.top_k == 0 || params.top_k > params.top_n) {
        throw std::runtime_error(BuildErrorMessage(
            "RunBatchFloat32FromDeviceTopologyPersistent",
            "Persistent rerank requires 1 <= top_k <= top_n."));
    }

    const std::size_t budget_top_n = std::min(params.top_n, topology_result.candidate_capacity);
    if (budget_top_n == 0) {
        return RerankBatchResult{};
    }

    const UploadedFloatQueries uploaded_queries = UploadFloatQueries(queries);
    PersistentBamRerankRunResult persistent_result =
        RunPersistentBamRerankFloat32(resources, uploaded_queries.device_queries, topology_result,
                                      num_queries, params);

    RerankExactParams finalize_params = params;
    finalize_params.top_n = persistent_result.result_top_n;
    RerankBatchResult result = FinalizeFusedRerankResults(
        num_queries, finalize_params, std::move(persistent_result.final_distances),
        std::move(persistent_result.final_node_ids), persistent_result.exact_count,
        persistent_result.exact_count, persistent_result.bound_filtered_count, 0.0,
        persistent_result.kernel_ms, 0.0, uploaded_queries.upload_ms, 0.0, 0.0, 0.0,
        total_begin, true, out_profile);
    result.profile.requested_top_n = params.top_n;
    result.profile.effective_top_n = budget_top_n;
    result.profile.rerank_tile_size = params.rank_tile_size;
    result.profile.rerank_bam_direct_batches = 1;
    result.profile.learned_stop_model_ms = persistent_result.learned_stop_model_ms;
    result.profile.learned_stop_checkpoint_bookkeeping_ms =
        persistent_result.learned_stop_checkpoint_bookkeeping_ms;
    result.profile.learned_stop_topk_churn_ms =
        persistent_result.learned_stop_topk_churn_ms;
    result.profile.learned_stop_next_window_scan_ms =
        persistent_result.learned_stop_next_window_scan_ms;
    result.profile.learned_stop_logit_eval_ms =
        persistent_result.learned_stop_logit_eval_ms;
    result.profile.rerank_query_block_ms = persistent_result.query_block_ms;
    result.profile.learned_stop_checkpoints = persistent_result.learned_stop_checkpoints;
    result.profile.learned_stop_queries = persistent_result.learned_stop_queries;
    result.profile.learned_stop_prefix_sum = persistent_result.learned_stop_prefix_sum;
    result.profile.learned_stop_prefix_min = persistent_result.learned_stop_prefix_min;
    result.profile.learned_stop_prefix_max = persistent_result.learned_stop_prefix_max;
    if (out_profile != nullptr) {
        out_profile->requested_top_n += params.top_n - persistent_result.result_top_n;
        out_profile->effective_top_n += budget_top_n - persistent_result.result_top_n;
        out_profile->rerank_tile_size += result.profile.rerank_tile_size;
        out_profile->rerank_bam_direct_batches += 1;
        out_profile->learned_stop_model_ms += result.profile.learned_stop_model_ms;
        out_profile->learned_stop_checkpoint_bookkeeping_ms +=
            result.profile.learned_stop_checkpoint_bookkeeping_ms;
        out_profile->learned_stop_topk_churn_ms +=
            result.profile.learned_stop_topk_churn_ms;
        out_profile->learned_stop_next_window_scan_ms +=
            result.profile.learned_stop_next_window_scan_ms;
        out_profile->learned_stop_logit_eval_ms +=
            result.profile.learned_stop_logit_eval_ms;
        out_profile->learned_stop_checkpoints += result.profile.learned_stop_checkpoints;
        out_profile->learned_stop_queries += result.profile.learned_stop_queries;
        out_profile->learned_stop_prefix_sum += result.profile.learned_stop_prefix_sum;
        if (result.profile.learned_stop_prefix_max != 0U ||
            result.profile.learned_stop_prefix_min != 0U) {
            if (out_profile->learned_stop_prefix_max == 0U &&
                out_profile->learned_stop_prefix_min == 0U) {
                out_profile->learned_stop_prefix_min = result.profile.learned_stop_prefix_min;
                out_profile->learned_stop_prefix_max = result.profile.learned_stop_prefix_max;
            } else {
                out_profile->learned_stop_prefix_min = std::min(
                    out_profile->learned_stop_prefix_min, result.profile.learned_stop_prefix_min);
                out_profile->learned_stop_prefix_max = std::max(
                    out_profile->learned_stop_prefix_max, result.profile.learned_stop_prefix_max);
            }
        }
    }
    return result;
}

Pq2RefineBatchResult RunPq2RefineBatchDevice(const SearchResources& resources,
                                             const DeviceTopologyBatchResult& topology_result,
                                             const std::vector<float>& queries,
                                             std::size_t num_queries,
                                             std::size_t refine_top_l) {
    if (!resources.has_pq2_index()) {
        throw std::runtime_error(
            BuildErrorMessage("RunPq2RefineBatchDevice", "PQ2 index must be loaded."));
    }
    if (topology_result.num_queries != num_queries) {
        throw std::runtime_error(BuildErrorMessage("RunPq2RefineBatchDevice",
                                                   "topology_result size mismatch."));
    }
    if (num_queries == 0 || refine_top_l == 0) {
        return {};
    }

    const PqIndex& pq2_index = resources.pq2_index();
    if (!pq2_index.codes_are_mapped()) {
        throw std::runtime_error(BuildErrorMessage("RunPq2RefineBatchDevice",
                                                   "PQ2 must use mapped host codes."));
    }
    if (pq2_index.host().num_chunks > kMaxPq2RefineChunks) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPq2RefineBatchDevice", "PQ2 num_chunks exceeds kernel specialization limit."));
    }
    if (refine_top_l > topology_result.candidate_capacity) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPq2RefineBatchDevice", "refine_top_l exceeds topology candidate capacity."));
    }
    if (refine_top_l > kMaxRerankSortCandidates) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPq2RefineBatchDevice", "refine_top_l exceeds device bitonic sort limit."));
    }

    const PqDistanceOracle pq2_oracle =
        PqDistanceOracle::FromFloatQueriesDeviceOnly(pq2_index, queries, num_queries);
    const std::size_t query_stride = pq2_index.host().num_chunks * kNumPqCentroids;
    const std::size_t padded_top_l = NextPowerOfTwo(refine_top_l);

    Pq2RefineBatchResult result;
    result.num_queries = num_queries;
    result.candidate_capacity = refine_top_l;
    result.pq_profile = pq2_oracle.query_tables().profile();
    result.residual_refine = resources.pq2_is_residual_refine();
    result.candidate_buffer =
        CudaBuffer<detail::DeviceTopologyCandidate>::Allocate(num_queries * refine_top_l);

    {
        const std::vector<DeviceTopologySearchStats> host_stats =
            topology_result.stats_buffer.CopyToHost();
        bool has_input = false;
        for (const auto& stats : host_stats) {
            const std::size_t input_candidates =
                std::min<std::size_t>(stats.valid_candidates, refine_top_l);
            result.input_valid_candidates += input_candidates;
            if (!has_input) {
                result.input_valid_candidates_min = input_candidates;
                result.input_valid_candidates_max = input_candidates;
                has_input = true;
            } else {
                result.input_valid_candidates_min =
                    std::min(result.input_valid_candidates_min, input_candidates);
                result.input_valid_candidates_max =
                    std::max(result.input_valid_candidates_max, input_candidates);
            }
        }
    }

    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin), "cudaEventRecord");
    if (resources.pq2_is_residual_refine()) {
        if (!resources.has_pq_index()) {
            throw std::runtime_error(BuildErrorMessage(
                "RunPq2RefineBatchDevice",
                "Base PQ index must be loaded for residual PQ2 refine."));
        }
        pq2_residual_refine_bitonic_sort_kernel<<<num_queries, kPq2RefineThreads>>>(
            topology_result.candidate_buffer.get(), topology_result.stats_buffer.get(),
            topology_result.candidate_capacity, refine_top_l, padded_top_l,
            resources.pq_index().device_codes(), pq2_index.device_codes(),
            pq2_index.host().num_chunks, pq2_oracle.query_tables().device_tables().get(),
            query_stride, resources.pq2_cross_terms().get(), result.candidate_buffer.get());
        ThrowIfCudaError(cudaGetLastError(), "pq2_residual_refine_bitonic_sort_kernel");
    } else {
        pq2_refine_bitonic_sort_kernel<<<num_queries, kPq2RefineThreads>>>(
            topology_result.candidate_buffer.get(), topology_result.stats_buffer.get(),
            topology_result.candidate_capacity, refine_top_l, padded_top_l, pq2_index.device_codes(),
            pq2_index.host().num_chunks, pq2_oracle.query_tables().device_tables().get(),
            query_stride, result.candidate_buffer.get());
        ThrowIfCudaError(cudaGetLastError(), "pq2_refine_bitonic_sort_kernel");
    }
    ThrowIfCudaError(cudaEventRecord(kernel_end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);
    result.kernel_ms = static_cast<double>(kernel_ms);
    return result;
}

Pq2RefineBatchResult RunPq2RefineBatchDevice(const SearchResources& resources,
                                             const DeviceTopologyBatchResult& topology_result,
                                             const CudaBuffer<float>& device_queries,
                                             std::size_t num_queries,
                                             std::size_t refine_top_l,
                                             cudaStream_t stream) {
    if (!resources.has_pq2_index()) {
        throw std::runtime_error(
            BuildErrorMessage("RunPq2RefineBatchDevice", "PQ2 index must be loaded."));
    }
    if (topology_result.num_queries != num_queries) {
        throw std::runtime_error(BuildErrorMessage("RunPq2RefineBatchDevice",
                                                   "topology_result size mismatch."));
    }
    if (num_queries == 0 || refine_top_l == 0) {
        return {};
    }

    const PqIndex& pq2_index = resources.pq2_index();
    if (!pq2_index.codes_are_mapped()) {
        throw std::runtime_error(BuildErrorMessage("RunPq2RefineBatchDevice",
                                                   "PQ2 must use mapped host codes."));
    }
    if (pq2_index.host().num_chunks > kMaxPq2RefineChunks) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPq2RefineBatchDevice", "PQ2 num_chunks exceeds kernel specialization limit."));
    }
    if (refine_top_l > topology_result.candidate_capacity) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPq2RefineBatchDevice", "refine_top_l exceeds topology candidate capacity."));
    }
    if (refine_top_l > kMaxRerankSortCandidates) {
        throw std::runtime_error(BuildErrorMessage(
            "RunPq2RefineBatchDevice", "refine_top_l exceeds device bitonic sort limit."));
    }

    const PqDistanceOracle pq2_oracle(
        pq2_index, PqQueryDistanceTables::FromFloatQueriesDeviceBufferAsync(
                       pq2_index, device_queries, num_queries, stream));
    const std::size_t query_stride = pq2_index.host().num_chunks * kNumPqCentroids;
    const std::size_t padded_top_l = NextPowerOfTwo(refine_top_l);

    Pq2RefineBatchResult result;
    result.num_queries = num_queries;
    result.candidate_capacity = refine_top_l;
    result.pq_profile = pq2_oracle.query_tables().profile();
    result.residual_refine = resources.pq2_is_residual_refine();
    result.candidate_buffer =
        CudaBuffer<detail::DeviceTopologyCandidate>::Allocate(num_queries * refine_top_l);

    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin, stream), "cudaEventRecord");
    if (resources.pq2_is_residual_refine()) {
        if (!resources.has_pq_index()) {
            throw std::runtime_error(BuildErrorMessage(
                "RunPq2RefineBatchDevice",
                "Base PQ index must be loaded for residual PQ2 refine."));
        }
        pq2_residual_refine_bitonic_sort_kernel<<<num_queries, kPq2RefineThreads, 0, stream>>>(
            topology_result.candidate_buffer.get(), topology_result.stats_buffer.get(),
            topology_result.candidate_capacity, refine_top_l, padded_top_l,
            resources.pq_index().device_codes(), pq2_index.device_codes(),
            pq2_index.host().num_chunks, pq2_oracle.query_tables().device_tables().get(),
            query_stride, resources.pq2_cross_terms().get(), result.candidate_buffer.get());
        ThrowIfCudaError(cudaGetLastError(), "pq2_residual_refine_bitonic_sort_kernel");
    } else {
        pq2_refine_bitonic_sort_kernel<<<num_queries, kPq2RefineThreads, 0, stream>>>(
            topology_result.candidate_buffer.get(), topology_result.stats_buffer.get(),
            topology_result.candidate_capacity, refine_top_l, padded_top_l,
            pq2_index.device_codes(), pq2_index.host().num_chunks,
            pq2_oracle.query_tables().device_tables().get(), query_stride,
            result.candidate_buffer.get());
        ThrowIfCudaError(cudaGetLastError(), "pq2_refine_bitonic_sort_kernel");
    }
    ThrowIfCudaError(cudaEventRecord(kernel_end, stream), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);
    result.kernel_ms = static_cast<double>(kernel_ms);
    return result;
}

Pq2RefineBatchResult RunHpqRefineBatchDevice(const SearchResources& resources,
                                             const DeviceTopologyBatchResult& topology_result,
                                             const CudaBuffer<float>& device_queries,
                                             std::size_t num_queries,
                                             std::size_t refine_top_l,
                                             cudaStream_t stream) {
    if (!resources.has_hpq_index()) {
        throw std::runtime_error(
            BuildErrorMessage("RunHpqRefineBatchDevice", "HPQ index must be loaded."));
    }
    if (topology_result.num_queries != num_queries) {
        throw std::runtime_error(BuildErrorMessage(
            "RunHpqRefineBatchDevice", "topology_result size mismatch."));
    }
    if (num_queries == 0 || refine_top_l == 0) {
        return {};
    }

    const HpqIndex& hpq = resources.hpq_index();
    const PqIndex& base_index = hpq.base_index();
    const PqIndex& outlier_index = hpq.outlier_index();
    const std::size_t num_chunks = outlier_index.host().num_chunks;
    if (!outlier_index.codes_are_mapped()) {
        throw std::runtime_error(BuildErrorMessage(
            "RunHpqRefineBatchDevice", "HPQ hybrid codes must use mapped host memory."));
    }
    if (num_chunks > kMaxPq2RefineChunks ||
        hpq.selector_stride_bytes() > kMaxHpqSelectorBytes) {
        throw std::runtime_error(BuildErrorMessage(
            "RunHpqRefineBatchDevice", "HPQ layout exceeds kernel limits."));
    }
    if (refine_top_l > topology_result.candidate_capacity ||
        refine_top_l > kMaxRerankSortCandidates) {
        throw std::runtime_error(BuildErrorMessage(
            "RunHpqRefineBatchDevice", "HPQ refine size exceeds candidate limits."));
    }

    PqQueryDistanceTables base_tables =
        PqQueryDistanceTables::FromFloatQueriesDeviceBufferAsync(
            base_index, device_queries, num_queries, stream);
    PqQueryDistanceTables outlier_tables =
        PqQueryDistanceTables::FromFloatQueriesDeviceBufferAsync(
            outlier_index, device_queries, num_queries, stream);
    const std::size_t query_stride = num_chunks * kNumPqCentroids;
    const std::size_t padded_top_l = NextPowerOfTwo(refine_top_l);

    Pq2RefineBatchResult result;
    result.num_queries = num_queries;
    result.candidate_capacity = refine_top_l;
    result.residual_refine = false;
    result.candidate_buffer =
        CudaBuffer<detail::DeviceTopologyCandidate>::Allocate(num_queries * refine_top_l);

    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin, stream), "cudaEventRecord");
    hpq_refine_bitonic_sort_kernel<<<num_queries, kPq2RefineThreads, 0, stream>>>(
        topology_result.candidate_buffer.get(), topology_result.stats_buffer.get(),
        topology_result.candidate_capacity, refine_top_l, padded_top_l,
        outlier_index.device_codes(), hpq.device_selector_bits(),
        hpq.selector_stride_bytes(), num_chunks, base_tables.device_tables().get(),
        outlier_tables.device_tables().get(), query_stride, result.candidate_buffer.get());
    ThrowIfCudaError(cudaGetLastError(), "hpq_refine_bitonic_sort_kernel");
    ThrowIfCudaError(cudaEventRecord(kernel_end, stream), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);
    result.kernel_ms = static_cast<double>(kernel_ms);
    return result;
}

RerankBatchResult RunBatchFloat32FromDeviceTopology(
    const SearchResources& resources,
    const DeviceTopologyBatchResult& topology_result,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const RerankExactParams& params,
    RerankBatchProfile* out_profile) {
    if (!resources.has_vector_store()) {
        throw std::runtime_error(BuildErrorMessage(
            "RunBatchFloat32FromDeviceTopology", "Vector store must be loaded before rerank."));
    }
    const auto& header = resources.vector_store_header();
    if (queries.size() != num_queries * header.dim) {
        throw std::runtime_error(BuildErrorMessage(
            "RunBatchFloat32FromDeviceTopology",
            "query buffer size must equal num_queries * dim."));
    }
    if (params.top_n > kMaxRerankSortCandidates) {
        // Larger top_n falls back to host-side final sort in FinalizeFusedRerankResults.
    }
    if (topology_result.num_queries != num_queries) {
        throw std::runtime_error(BuildErrorMessage(
            "RunBatchFloat32FromDeviceTopology", "topology_result size mismatch."));
    }
    if (!resources.vector_page_provider().SupportsDeviceReads()) {
        throw std::runtime_error(BuildErrorMessage("RunBatchFloat32FromDeviceTopology",
                                                   "Fused rerank requires device-direct reads."));
    }
    if (num_queries == 0 || params.top_n == 0) {
        return RerankBatchResult{};
    }
    if (params.use_pq2_bound_filter) {
        if (!params.use_pq2_refine) {
            throw std::runtime_error(BuildErrorMessage(
                "RunBatchFloat32FromDeviceTopology",
                "PQ2 bound filter requires PQ2 refine to be enabled."));
        }
        if (params.top_k == 0 || params.top_k > kMaxTrackedTopK) {
            throw std::runtime_error(BuildErrorMessage(
                "RunBatchFloat32FromDeviceTopology",
                "PQ2 bound filter requires 1 <= top_k <= 128."));
        }
        if (params.mode == RerankExecutionMode::kLinear) {
            throw std::runtime_error(BuildErrorMessage(
                "RunBatchFloat32FromDeviceTopology",
                "PQ2 bound filter requires a non-linear GPU rerank path."));
        }
        if (!resources.has_pq2_error_bounds()) {
            throw std::runtime_error(BuildErrorMessage(
                "RunBatchFloat32FromDeviceTopology",
                "PQ2 bound filter requires residual PQ2 error bounds."));
        }
    }
    if (params.use_learned_stop) {
        if (params.mode != RerankExecutionMode::kPersistent) {
            throw std::runtime_error(BuildErrorMessage(
                "RunBatchFloat32FromDeviceTopology",
                "learned rerank stop requires the persistent GPU rerank path."));
        }
        if (!params.use_pq2_bound_filter) {
            throw std::runtime_error(BuildErrorMessage(
                "RunBatchFloat32FromDeviceTopology",
                "learned rerank stop requires PQ34 refine + bound filter."));
        }
        if (params.learned_stop_model_path.empty()) {
            throw std::runtime_error(BuildErrorMessage(
                "RunBatchFloat32FromDeviceTopology",
                "learned rerank stop requires learned_stop_model_path."));
        }
    }

    const DeviceTopologyBatchResult* source_topology = &topology_result;
    DeviceTopologyBatchResult refined_topology_result;
    Pq2RefineBatchResult pq2_refine_result;
    const bool use_pq2_refine = params.use_pq2_refine;
    RerankExactParams exact_params = params;
    exact_params.use_pq2_refine = false;
    exact_params.pq2_refine_top_l = 0;
    if (use_pq2_refine) {
        const std::size_t refine_top_l = std::min(
            params.pq2_refine_top_l == 0 ? topology_result.candidate_capacity
                                         : params.pq2_refine_top_l,
            topology_result.candidate_capacity);
        pq2_refine_result = RunPq2RefineBatchDevice(resources, topology_result, queries,
                                                    num_queries, refine_top_l);
        refined_topology_result.num_queries = pq2_refine_result.num_queries;
        refined_topology_result.candidate_capacity = pq2_refine_result.candidate_capacity;
        refined_topology_result.candidate_buffer = std::move(pq2_refine_result.candidate_buffer);
        source_topology = &refined_topology_result;
    }

    RerankBatchResult result;
    switch (exact_params.mode) {
        case RerankExecutionMode::kLinear:
            result = RunBatchFloat32FromDeviceTopologyLinear(resources, *source_topology, queries,
                                                             num_queries, exact_params,
                                                             out_profile);
            break;
        case RerankExecutionMode::kTiled:
            result = RunBatchFloat32FromDeviceTopologyRankTiled(
                resources, *source_topology, queries, num_queries, exact_params,
                std::max<std::size_t>(1, exact_params.rank_tile_size), out_profile);
            break;
        case RerankExecutionMode::kPageByPage:
            result = RunBatchFloat32FromDeviceTopologyRankTiled(
                resources, *source_topology, queries, num_queries, exact_params, 1, out_profile);
            break;
        case RerankExecutionMode::kPersistent:
            result = RunBatchFloat32FromDeviceTopologyPersistent(resources, *source_topology,
                                                                queries, num_queries, exact_params,
                                                                out_profile);
            break;
    }
    if (use_pq2_refine) {
        result.profile.pq2_query_tables_ms += pq2_refine_result.pq_profile.total_ms;
        result.profile.pq2_query_upload_ms += pq2_refine_result.pq_profile.query_upload_ms;
        result.profile.pq2_query_zero_fill_ms += pq2_refine_result.pq_profile.zero_fill_ms;
        result.profile.pq2_query_table_kernel_ms += pq2_refine_result.pq_profile.kernel_ms;
        result.profile.pq2_query_table_download_ms +=
            pq2_refine_result.pq_profile.table_download_ms;
        result.profile.pq2_refine_kernel_ms += pq2_refine_result.kernel_ms;
        result.profile.pq2_refine_input_candidates +=
            pq2_refine_result.input_valid_candidates;
        result.profile.pq2_refine_input_candidates_min =
            pq2_refine_result.input_valid_candidates_min;
        result.profile.pq2_refine_input_candidates_max =
            pq2_refine_result.input_valid_candidates_max;
        result.profile.pq2_refine_top_l = pq2_refine_result.candidate_capacity;
        result.profile.pq2_refine_batches += 1;
        result.profile.pq2_residual_refine_batches +=
            pq2_refine_result.residual_refine ? 1 : 0;
        result.profile.prepare_ms +=
            pq2_refine_result.pq_profile.total_ms + pq2_refine_result.kernel_ms;
        result.profile.total_ms +=
            pq2_refine_result.pq_profile.total_ms + pq2_refine_result.kernel_ms;
        if (out_profile != nullptr) {
            out_profile->pq2_query_tables_ms += pq2_refine_result.pq_profile.total_ms;
            out_profile->pq2_query_upload_ms += pq2_refine_result.pq_profile.query_upload_ms;
            out_profile->pq2_query_zero_fill_ms += pq2_refine_result.pq_profile.zero_fill_ms;
            out_profile->pq2_query_table_kernel_ms += pq2_refine_result.pq_profile.kernel_ms;
            out_profile->pq2_query_table_download_ms +=
                pq2_refine_result.pq_profile.table_download_ms;
            out_profile->pq2_refine_kernel_ms += pq2_refine_result.kernel_ms;
            out_profile->pq2_refine_input_candidates +=
                pq2_refine_result.input_valid_candidates;
            const bool had_pq2_batches = out_profile->pq2_refine_batches != 0;
            bool has_dst_input = had_pq2_batches;
            MergeMinMax(pq2_refine_result.input_valid_candidates_min,
                        pq2_refine_result.input_valid_candidates_max, num_queries != 0,
                        &out_profile->pq2_refine_input_candidates_min,
                        &out_profile->pq2_refine_input_candidates_max, &has_dst_input);
            out_profile->pq2_refine_top_l += pq2_refine_result.candidate_capacity;
            out_profile->pq2_refine_batches += 1;
            out_profile->pq2_residual_refine_batches +=
                pq2_refine_result.residual_refine ? 1 : 0;
            out_profile->prepare_ms +=
                pq2_refine_result.pq_profile.total_ms + pq2_refine_result.kernel_ms;
            out_profile->total_ms +=
                pq2_refine_result.pq_profile.total_ms + pq2_refine_result.kernel_ms;
        }
    }
    return result;
}

}  // namespace detail

RerankBatchResult RerankExact::RunBatchFloat32(
    const SearchResources& resources,
    const std::vector<TopologySearchResult>& topology_results,
    const std::vector<float>& queries,
    std::size_t num_queries,
    const RerankExactParams& params,
    RerankBatchProfile* out_profile) {
    return RunBatchImpl(resources, topology_results, queries, num_queries, params,
                        "RerankExact::RunBatchFloat32", out_profile);
}

RerankBatchResult RerankExact::RunBatchUint8(
    const SearchResources& resources,
    const std::vector<TopologySearchResult>& topology_results,
    const std::vector<std::uint8_t>& queries,
    std::size_t num_queries,
    const RerankExactParams& params,
    RerankBatchProfile* out_profile) {
    return RunBatchImpl(resources, topology_results, queries, num_queries, params,
                        "RerankExact::RunBatchUint8", out_profile);
}

RerankBatchResult RerankExact::RunBatchInt8(
    const SearchResources& resources,
    const std::vector<TopologySearchResult>& topology_results,
    const std::vector<std::int8_t>& queries,
    std::size_t num_queries,
    const RerankExactParams& params,
    RerankBatchProfile* out_profile) {
    return RunBatchImpl(resources, topology_results, queries, num_queries, params,
                        "RerankExact::RunBatchInt8", out_profile);
}

}  // namespace topoanns
