#pragma once

// #include <cuda.h>

__global__ void findMinIndicesKernel(float* values, int c, int num, int* minIndices);

__global__ void findMinIndicesKernel(int* d_min_coarse_indices, float* values, int c, int num, int* d_enter_cluster, int* d_sizes);

std::vector<int> findMinIndices(const float* values, int c, int num);

void deviceFindMinIndices(float* values, int c, int num, int* d_min_indices);

void deviceFindMinIndices(int* d_min_coarse_indices, float* d_values, int c, int num, int* d_enter_cluster, int* d_sizes);