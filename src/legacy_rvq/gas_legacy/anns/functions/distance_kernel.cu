#include "check.h"
#include "distance_kernel.cuh"
#include "../common.h"

#include <cublas_v2.h>
#include <stdio.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 128
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)
#define FULL_MASK 0xffffffff

__device__ inline val_t WarpReduce(val_t val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

template <num_t RowTileSize>
__global__ void L2NormKernel(val_t *data, num_t n, num_t dim, val_t *result) {
    extern __shared__ val_t smem[]; // warps * RowTileSize elements

    int warps = blockDim.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    bool last_row_tile = (blockIdx.x == (gridDim.x - 1));
    size_t row_start = (size_t)(blockIdx.x) * RowTileSize; //*

    val_t row_norm[RowTileSize];

    if (last_row_tile) {
        for (int row = 0; row < n - row_start; row++) {
            row_norm[0] = 0;
            for (size_t col = threadIdx.x; col < dim; col += blockDim.x) {
                val_t val = data[(row_start + row) * dim + col];
                row_norm[0] += (val * val);
            }
            row_norm[0] = WarpReduce(row_norm[0]);
            if (lane_id == 0) {
                smem[row * warps + warp_id] = row_norm[0];
            }
        }
    } else {
        // RowTileSize rows per thread
        val_t tmp[RowTileSize];
#pragma unroll
        for (int row = 0; row < RowTileSize; row++) {
            row_norm[row] = 0;
        }

        for (size_t col = threadIdx.x; col < dim; col += blockDim.x) {
#pragma unroll
            for (int row = 0; row < RowTileSize; row++) {
                tmp[row] = data[(row_start + row) * dim + col];
            }
#pragma unroll
            for (int row = 0; row < RowTileSize; row++) {
                tmp[row] = tmp[row] * tmp[row];
            }
#pragma unroll
            for (int row = 0; row < RowTileSize; row++) {
                row_norm[row] += tmp[row];
            }
        }
        // warp reduce
#pragma unroll
        for (int row = 0; row < RowTileSize; row++) {
            row_norm[row] = WarpReduce(row_norm[row]);
        }
        // store result of each warp into shard memory
        if (lane_id == 0) {
#pragma unroll
            for (int row = 0; row < RowTileSize; row++) {
                smem[row * warps + warp_id] = row_norm[row];
            }
        }
    }
    __syncthreads();
    //* sum across warps
    if (warp_id == 0) {
        // load results in shared memory to register
#pragma unroll
        for (int row = 0; row < RowTileSize; row++) {
            row_norm[row] = lane_id < warps ? smem[row * warps + lane_id] : 0;
        }
#pragma unroll
        for (int row = 0; row < RowTileSize; row++) {
            row_norm[row] = WarpReduce(row_norm[row]);
        }
        // write out answer
        if (lane_id == 0) {
#pragma unroll
            for (int row = 0; row < RowTileSize; row++) {
                size_t out_col = row_start + row;
                if (last_row_tile) {
                    if (out_col < n) {
                        result[out_col] = row_norm[row];
                    }
                } else {
                    result[out_col] = row_norm[row];
                }
            }
        }
    }
}
//* A, B, C are stored in row major manner
//* A(m * k) * B(n * k) = C (m * n)
void RunMatrixMult(val_t *d_A, val_t *d_B, val_t *d_C, num_t m, num_t n,
                   num_t k) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = -2.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)m,
                               (int)k, &alpha, d_B, CUDA_R_32F, (int)k, d_A,
                               CUDA_R_32F, (int)k, &beta, d_C, CUDA_R_32F,
                               (int)n));
    CUBLAS_CHECK(cublasDestroy(handle));
}

template <num_t kRowsPerBlock, num_t kRowUnroll, num_t kColLoad>
__global__ void SumAlongCloumsKernel(val_t *input, val_t *output, num_t m,
                                     num_t n) {
    size_t row_start = (size_t)blockIdx.x * kRowsPerBlock; // start row of block
    size_t row_end = row_start + kRowsPerBlock;
    size_t col_start =
        (size_t)blockIdx.y * (blockDim.x * kColLoad); // start column of block

    bool end_row = (blockIdx.x == gridDim.x - 1);
    bool end_col = (blockIdx.y == gridDim.y - 1);

    if (end_row) {
        if (m % kRowsPerBlock == 0) {
            end_row = false;
        }
    }
    if (end_col) {
        for (size_t col = col_start + threadIdx.x; col < n; col += blockDim.x) {
            val_t val = input[col];
            if (end_row) {
                for (size_t row = row_start; row < m; row++) {
                    output[row * n + col] += val; // sum
                }
            } else {
                for (size_t row = row_start; row < row_end; row += kRowUnroll) {
#pragma unroll
                    for (int i = 0; i < kRowUnroll; i++) {
                        output[(row + i) * n + col] += val;
                    }
                }
            }
        }
    } else {
        size_t col = col_start + threadIdx.x; // stride = blockDim.x
        val_t vals[kColLoad];
#pragma unroll
        for (int i = 0; i < kColLoad; i++) {
            vals[i] = input[col + (i * blockDim.x)];
        }
        if (end_row) {
            for (size_t row = row_start; row < m; row++) {
#pragma unroll
                for (int i = 0; i < kColLoad; i++) {
                    output[row * n + col + (i * blockDim.x)] += vals[i];
                }
            }
        } else {
            for (size_t row = row_start; row < row_end; row += kRowUnroll) {
#pragma unroll
                for (int i = 0; i < kRowUnroll; i++) {
#pragma unroll
                    for (int j = 0; j < kColLoad; j++) {
                        output[(row + i) * n + col + (j * blockDim.x)] +=
                            vals[j];
                    }
                }
            }
        }
    }
}

__global__ void SumAlongRowsKernel(val_t *input, val_t *output, num_t m,
                                   num_t n) {
    __shared__ val_t sval;
    size_t row = blockIdx.x;
    if (threadIdx.x == 0) {
        sval = input[row];
    }
    __syncthreads();
    val_t val = sval;
    for (size_t col = threadIdx.x; col < n; col += blockDim.x) {
        output[row * n + col] += val;
    }
}

//* output: (m * n)
void RunSumAlongColumns(val_t *input, val_t *output, num_t m, num_t n) {
    num_t threads_per_block = 256;
    constexpr num_t k_row_unroll = 4;
    constexpr num_t k_rows_per_block = k_row_unroll * 4;
    constexpr num_t k_col_load = 4;

    num_t row_tiles = (m + k_rows_per_block - 1) / k_rows_per_block;
    num_t col_tiles = (n + threads_per_block * k_col_load - 1) /
                      (threads_per_block * k_col_load);
    auto block = dim3(threads_per_block);
    auto grid = dim3(row_tiles, col_tiles);

    SumAlongCloumsKernel<k_rows_per_block, k_row_unroll, k_col_load>
        <<<grid, block>>>(input, output, m, n);
    CUDA_SYNC_CHECK();
}

void RunSumAlongRows(val_t *input, val_t *output, num_t m, num_t n) {
    num_t threads_per_block = std::min(n, (num_t)1024);
    auto grid = dim3(m);
    auto block = dim3(threads_per_block);

    SumAlongRowsKernel<<<grid, block>>>(input, output, m, n);
    CUDA_SYNC_CHECK();
}

// x^2+y^2-2xy
void deviceQueryToBaseDistance(val_t *d_base_data, num_t base_num, val_t *d_query_data, num_t query_num, num_t dim, val_t *dis_matrix, const num_t ChunkSize) {
    // printf("[Info] distance calculation on %d centroids and %d quries\n", base_num, query_num);
    // compute queries
    Timer queryT;
    constexpr num_t row_tile_size = 8;
    int block_size = BLOCK_SIZE;
    //* query L2Norm
    //queryT.Start();
    val_t *d_query_norms;
    CUDA_CHECK(cudaMalloc((void **)&d_query_norms, sizeof(val_t) * query_num));
    //queryT.Stop();
    //std::cout<<"[RVQ] 1: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
    int blocks = (query_num + row_tile_size - 1) / row_tile_size;
    size_t smem_bytes = sizeof(val_t) * row_tile_size * WARPS_PER_BLOCK;
    //queryT.Start();
    L2NormKernel<row_tile_size><<<blocks, block_size, smem_bytes, 0>>>(
        d_query_data, query_num, dim, d_query_norms);
    CUDA_SYNC_CHECK();
    //queryT.Stop();
    //std::cout<<"[RVQ] 7: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;

    // TODO: result matrix is too large, so we split the base data to multiple
    // chunks
    // printf("Base data chunk size = %d\n", ChunkSize);
    num_t chunks = (base_num + ChunkSize - 1) / ChunkSize;
    // printf("Chunks = %d\n", chunks);
    for (num_t chunk = 0; chunk < chunks; chunk++) {
        // printf("Start %d iteration...\n", chunk);
        size_t base_start = chunk * ChunkSize;
        size_t base_size =
            ((base_start + ChunkSize < base_num) ? ChunkSize
                                                 : base_num - base_start);

        //* base data L2Norm
        //queryT.Start();
        val_t *d_base_norms;
        CUDA_CHECK(
            cudaMalloc((void **)&d_base_norms, sizeof(val_t) * base_size));
        //queryT.Stop();
        //std::cout<<"[RVQ] 2: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
        blocks = (base_size + row_tile_size - 1) / row_tile_size;
        smem_bytes = sizeof(val_t) * row_tile_size * WARPS_PER_BLOCK;
        //queryT.Start();
        L2NormKernel<row_tile_size><<<blocks, block_size, smem_bytes, 0>>>(
            d_base_data, base_size, dim, d_base_norms);
        CUDA_SYNC_CHECK();
        //queryT.Stop();
        //std::cout<<"[RVQ] 8: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
        // printf("Debug 1 ...\n");
        //* ====================
        //* A(query_num * dim) * B(base_size * dim) => C(query_num * base_size)
        // queryT.Start();
        // val_t *d_mults;
        // CUDA_CHECK(cudaMalloc((void **)&d_mults,
        //                       sizeof(val_t) * query_num * base_size));
        // queryT.Stop();
        // std::cout<<"[RVQ] 3: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
        //queryT.Start();
        RunMatrixMult(d_query_data, d_base_data, dis_matrix, query_num, base_size,
                      dim);
        //queryT.Stop();
        //std::cout<<"[RVQ] 9: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
        //* ====================
        // printf("Debug 2 ...\n");
        //* add ||query||^2 along rows
        //queryT.Start();
        RunSumAlongRows(d_query_norms, dis_matrix, query_num, base_size);
        //queryT.Stop();
        //std::cout<<"[RVQ] 10: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
        // printf("Debug 3 ...\n");
        //* add ||base||^2 along columns
        //queryT.Start();
        RunSumAlongColumns(d_base_norms, dis_matrix, query_num, base_size);
        //queryT.Stop();
        //std::cout<<"[RVQ] 11: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
        // printf("Debug 4 ...\n");
        // * copy distance to host
        // queryT.Start();
        // CUDA_CHECK(cudaMemcpy(dis_matrix, d_mults,
        //                    sizeof(val_t) * base_size, cudaMemcpyDeviceToHost));
            //   printf("Debug 4.%d ...\n",i);
        // queryT.Stop();
        // std::cout<<"[RVQ] 4: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
        // printf("Debug 5 ...\n");
        //queryT.Start();
        CUDA_CHECK(cudaFree(d_base_norms));
        // CUDA_CHECK(cudaFree(d_mults));
        //queryT.Stop();
        //std::cout<<"[RVQ] 5: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
    }
    //queryT.Start();
    CUDA_CHECK(cudaFree(d_query_norms));
    //queryT.Stop();
    //std::cout<<"[RVQ] 6: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
}

// x^2+y^2-2xy
void queryToBaseDistance(val_t *base_data, num_t base_num, val_t *query_data, num_t query_num, num_t dim, val_t *dis_matrix, const num_t ChunkSize) {
    // printf("[Info] distance calculation on %d centroids and %d quries\n", base_num, query_num);
    // compute queries
    val_t *d_query_data;
    CUDA_CHECK(
        cudaMalloc((void **)&d_query_data, sizeof(val_t) * query_num * dim));
    CUDA_CHECK(cudaMemcpy(d_query_data, query_data,
                          sizeof(val_t) * query_num * dim,
                          cudaMemcpyHostToDevice));

    constexpr num_t row_tile_size = 8;
    int block_size = BLOCK_SIZE;
    //* query L2Norm
    val_t *d_query_norms;
    CUDA_CHECK(cudaMalloc((void **)&d_query_norms, sizeof(val_t) * query_num));
    int blocks = (query_num + row_tile_size - 1) / row_tile_size;
    size_t smem_bytes = sizeof(val_t) * row_tile_size * WARPS_PER_BLOCK;

    L2NormKernel<row_tile_size><<<blocks, block_size, smem_bytes, 0>>>(
        d_query_data, query_num, dim, d_query_norms);
    CUDA_SYNC_CHECK();

    // TODO: result matrix is too large, so we split the base data to multiple
    // chunks
    // printf("Base data chunk size = %d\n", ChunkSize);
    num_t chunks = (base_num + ChunkSize - 1) / ChunkSize;
    // printf("Chunks = %d\n", chunks);
    for (num_t chunk = 0; chunk < chunks; chunk++) {
        // printf("Start %d iteration...\n", chunk);
        size_t base_start = chunk * ChunkSize;
        size_t base_size =
            ((base_start + ChunkSize < base_num) ? ChunkSize
                                                 : base_num - base_start);

        // copy data to device
        val_t *d_base_data;
        CUDA_CHECK(cudaMalloc((void **)&d_base_data,
                              sizeof(val_t) * (base_size * dim)));
        CUDA_CHECK(cudaMemcpy(d_base_data, base_data + (base_start * dim),
                              sizeof(val_t) * (base_size * dim),
                              cudaMemcpyHostToDevice));

        //* base data L2Norm
        val_t *d_base_norms;
        CUDA_CHECK(
            cudaMalloc((void **)&d_base_norms, sizeof(val_t) * base_size));
        blocks = (base_size + row_tile_size - 1) / row_tile_size;
        smem_bytes = sizeof(val_t) * row_tile_size * WARPS_PER_BLOCK;

        L2NormKernel<row_tile_size><<<blocks, block_size, smem_bytes, 0>>>(
            d_base_data, base_size, dim, d_base_norms);
        CUDA_SYNC_CHECK();
        // printf("Debug 1 ...\n");
        //* ====================
        //* A(query_num * dim) * B(base_size * dim) => C(query_num * base_size)
        val_t *d_mults;
        CUDA_CHECK(cudaMalloc((void **)&d_mults,
                              sizeof(val_t) * query_num * base_size));
        RunMatrixMult(d_query_data, d_base_data, d_mults, query_num, base_size,
                      dim);
        //* ====================
        // printf("Debug 2 ...\n");
        //* add ||query||^2 along rows
        RunSumAlongRows(d_query_norms, d_mults, query_num, base_size);
        // printf("Debug 3 ...\n");
        //* add ||base||^2 along columns
        RunSumAlongColumns(d_base_norms, d_mults, query_num, base_size);
        // printf("Debug 4 ...\n");
        //* copy distance to host
        for (num_t i = 0; i < query_num; i++) {
            CUDA_CHECK(
                cudaMemcpy(dis_matrix + (i * (size_t)base_num + base_start),
                           d_mults + (i * base_size),
                           sizeof(val_t) * base_size, cudaMemcpyDeviceToHost));
            //   printf("Debug 4.%d ...\n",i);
        }
        // printf("Debug 5 ...\n");
        CUDA_CHECK(cudaFree(d_base_norms));
        CUDA_CHECK(cudaFree(d_mults));
        CUDA_CHECK(cudaFree(d_base_data));
    }
    CUDA_CHECK(cudaFree(d_query_norms));
    CUDA_CHECK(cudaFree(d_query_data));
}

void SortOnDevice(DevicePair *data, num_t n) {
    thrust::device_vector<DevicePair> d_data(data, data + n);
    thrust::sort(d_data.begin(), d_data.end());
    thrust::copy(d_data.begin(), d_data.end(), data);
}

// Helper function to print matrices
void printMatrix(const char* name, const float* matrix, int rows, int cols) {
    std::cout << name << ":\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i * cols + j] << " ";
        }
        std::cout << "\n";
    }
}

// int main() {
//     // Number of rows (vectors) and dimensions
//     const int query_num = 10000;
//     const int base_num = 1024;
//     const int dim = 128;

//     // Define and initialize the query and base data
//     float *query_data = new float[query_num * dim];
//     float *base_data = new float[base_num * dim];

//     for (int i = 0; i < query_num * dim; i++) query_data[i] = static_cast<float>(1);
//     for (int i = 0; i < base_num * dim; i++) base_data[i] = static_cast<float>(0);

//     // Prepare a matrix to store the results
//     float* dis_matrix = new float[query_num * base_num];

//     // Call the function under test
//     int chunk = std::min(base_num, 1000000);
//     queryToBaseDistance(base_data, base_num, query_data, query_num, dim, dis_matrix, chunk);

//     // Print the results
//     // printMatrix("Distance Matrix", dis_matrix, query_num, base_num);

//     // Cleanup
//     delete[] dis_matrix;

//     return 0;
// }