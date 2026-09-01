#include <iostream>
#include <vector>
#include "selectMin1.cuh"
#include <float.h>

// CUDA kernel函数，用于在每个段中查找最小值的索引
// c : 每段的大小
// num : 段数
__global__ void findMinIndicesKernel(float* values, int c, int num, int* minIndices) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num) {
        int startIdx = idx * c;
        int endIdx = min((idx + 1) * c, num * c);
        float minVal = values[startIdx];
        int minIdx = 0;
        for (int i = startIdx + 1; i < endIdx; ++i) {
            if (values[i] < minVal) {
                minVal = values[i];
                minIdx = i - startIdx;
            }
        }
        minIndices[idx] = minIdx;
    }
    // int b_id = blockIdx.x;
    // int t_id = threadIdx.x;
    // int size_of_warp = 32;
    // float* start_loc = values + b_id * c;
    // int min_idx = 0;
    // float min_dis = FLT_MAX;
    // for (int i = 0; i < (c + size_of_warp - 1) / size_of_warp; i++){
    //     int unrollt_id = t_id + size_of_warp * i;
    //     int idx;
    //     float dis;
    //     if(unrollt_id < c){
    //         idx = unrollt_id;
    //         dis = start_loc[unrollt_id ];
    //     }
    //     else{
    //         idx = c;
    //         dis = FLT_MAX;
    //     }
    //     for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    //         int temp_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
    //         float temp_dis = __shfl_down_sync(0xFFFFFFFF, dis, offset);

    //         if (temp_dis < dis) {
    //             idx = temp_idx;
    //             dis = temp_dis;
    //         }
    //     }
    //     if(t_id == 0){
    //         if(dis < min_dis){
    //             min_dis = dis;
    //             min_idx = idx;
    //         }
    //     }
    // }
    // if(t_id == 0){
    //     minIndices[b_id] = min_idx;
    // }
}

// CUDA kernel函数，用于在每个段中查找最小值并且索引内点数不为0的索引
// c : 每段的大小
// num : 段数
__global__ void findMinIndicesKernel(int* d_min_coarse_indices,float* values, int c, int num, int* d_enter_cluster, int* d_sizes) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < num) {
        int coarse_indices = d_min_coarse_indices[idx];
        int startIdx = idx * c;
        int endIdx = min((idx + 1) * c, num * c);
        float minVal = FLT_MAX;
        int minIdx = 0;
        for (int i = startIdx; i < endIdx; ++i) {
            if (values[i] < minVal && d_sizes[coarse_indices * c + i - startIdx] > 0) {
                minVal = values[i];
                minIdx = i - startIdx;
            }
        }
        d_enter_cluster[idx] = coarse_indices * c + minIdx;
    }
    // int b_id = blockIdx.x;
    // int t_id = threadIdx.x;
    // int coarse_indices = d_min_coarse_indices[b_id];
    // int size_of_warp = 32;
    // float* start_loc = values + b_id * c;
    // int min_idx = 0;
    // float min_dis = FLT_MAX;
    // for (int i = 0; i < (c + size_of_warp - 1) / size_of_warp; i++){
    //     int unrollt_id = t_id + size_of_warp * i;
    //     int idx;
    //     float dis;
    //     if(unrollt_id < c && d_sizes[coarse_indices * c + unrollt_id] > 0){
    //         idx = unrollt_id;
    //         dis = start_loc[unrollt_id ];
    //     }
    //     else{
    //         idx = c;
    //         dis = FLT_MAX;
    //     }
    //     for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    //         int temp_idx = __shfl_down_sync(0xFFFFFFFF, idx, offset);
    //         float temp_dis = __shfl_down_sync(0xFFFFFFFF, dis, offset);

    //         if (temp_dis < dis) {
    //             idx = temp_idx;
    //             dis = temp_dis;
    //         }
    //     }
    //     if(t_id == 0){
    //         if(dis < min_dis){
    //             min_dis = dis;
    //             min_idx = idx;
    //         }
    //     }
    // }
    // if(t_id == 0){
    //     d_enter_cluster[b_id] = coarse_indices * c + min_idx;
    // }
}

// 调用CUDA kernel函数并将结果复制回主机内存
std::vector<int> findMinIndices(const float* values, int c, int num) {
    int totalNum = num;
    std::vector<int> minIndices(totalNum, 0); // 初始化为0

    float* d_values;
    int* d_minIndices;
    cudaMalloc((void**)&d_values, sizeof(float) * c * num);
    cudaMalloc((void**)&d_minIndices, sizeof(int) * totalNum);

    cudaMemcpy(d_values, values, sizeof(float) * c * num, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (totalNum + blockSize - 1) / blockSize;

    findMinIndicesKernel<<<numBlocks, blockSize>>>(d_values, c, num, d_minIndices);

    cudaMemcpy(minIndices.data(), d_minIndices, sizeof(int) * totalNum, cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_minIndices);

    return minIndices;
}

// 调用CUDA kernel函数并将最终结果存放在GPU显存
void deviceFindMinIndices(int* d_min_coarse_indices, float* d_values, int c, int num, int* d_enter_cluster, int* d_sizes) {
    int totalNum = num;

    int blockSize = 256;
    int numBlocks = (totalNum + blockSize - 1) / blockSize;

    findMinIndicesKernel<<<numBlocks, blockSize>>>(d_min_coarse_indices, d_values, c, num, d_enter_cluster, d_sizes);
}

// 调用CUDA kernel函数并将结果存放在GPU显存
void deviceFindMinIndices(float* d_values, int c, int num, int* d_min_indices) {
    int totalNum = num;

    int blockSize = 256;
    int numBlocks = (totalNum + blockSize - 1) / blockSize;

    findMinIndicesKernel<<<numBlocks, blockSize>>>(d_values, c, num, d_min_indices);
}

// int main() {
//     // 示例数据
//     std::vector<float> values = {1.2, 2.4, 3.1, 0.5, 2.0, 1.8, 3.5, 4.2, 0.9, 2.3, 235, 2.4};

//     // 每段的大小和段数
//     int c = 3;
//     int num = 4;

//     // 使用CUDA加速查找每段中的最小值索引
//     std::vector<int> minIndices = findMinIndices(values.data(), c, num);

//     // 打印结果
//     std::cout << "Minimum indices in each segment:" << std::endl;
//     for (int i = 0; i < num; ++i) {
//         std::cout << "Segment " << i << ": " << minIndices[i] << std::endl;
//     }

//     return 0;
// }