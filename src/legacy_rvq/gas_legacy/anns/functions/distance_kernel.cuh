#pragma once

#include "../common.h"

#ifndef __host__
#define __host__
#endif

#ifndef __device__
#define __device__
#endif

#ifndef __global__
#define __global__
#endif

// 声明 queryToBaseDistance 函数
void queryToBaseDistance(val_t *base_data, num_t base_num, val_t *query_data,
                         num_t query_num, num_t dim, val_t *dis_matrix,
                         const num_t ChunkSize = 100000);

void deviceQueryToBaseDistance(val_t *d_base_data, num_t base_num,
                        val_t *d_query_data, num_t query_num, num_t dim,
                        val_t *dis_matrix, const num_t ChunkSize);

struct DevicePair {
    val_t val;
    idx_t idx;
    __host__ __device__ DevicePair() {}
    __host__ __device__ DevicePair(val_t v, idx_t i) : val(v), idx(i) {}
    __host__ __device__ bool operator<(const DevicePair &d) const {
        if (val == d.val)
            return idx < d.idx;
        else
            return val < d.val;
    }
};

// less is first
void SortOnDevice(DevicePair *data, num_t n);
