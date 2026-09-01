#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace topoanns::detail {

constexpr unsigned int kExactDistanceFullWarpMask = 0xffffffffU;
constexpr std::size_t kExactDistanceWarpSize = 32;

template <std::size_t kDim>
__device__ __forceinline__ float WarpFloat32SquaredL2(const float* vector,
                                                      const float* query,
                                                      std::size_t lane_id) {
    float distance = 0.0f;
#pragma unroll
    for (std::size_t dim_idx = lane_id; dim_idx < kDim;
         dim_idx += kExactDistanceWarpSize) {
        const float diff = vector[dim_idx] - query[dim_idx];
        distance += diff * diff;
    }
    distance += __shfl_down_sync(kExactDistanceFullWarpMask, distance, 16);
    distance += __shfl_down_sync(kExactDistanceFullWarpMask, distance, 8);
    distance += __shfl_down_sync(kExactDistanceFullWarpMask, distance, 4);
    distance += __shfl_down_sync(kExactDistanceFullWarpMask, distance, 2);
    distance += __shfl_down_sync(kExactDistanceFullWarpMask, distance, 1);
    return distance;
}

}  // namespace topoanns::detail
