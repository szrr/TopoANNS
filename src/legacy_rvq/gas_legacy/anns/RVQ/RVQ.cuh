/**
 * @author szr
 * @date 2024/5/24
 * @brief two-layer k-means using RVQ method
 *
 * **/

#pragma once
#include <iostream>
#include <vector>
// #include <mkl_cblas.h>
// #include <mkl.h>
// #include <mkl_service.h>
#include "../common.h"
#include "../functions/check.h"

float kmeans(float* trainData, int numTrainData, int dim, float* codebook, int numCentroids, int* assign);
void rand_perm(int* perm, size_t n, int64_t seed);
void fillWithRandom(float* data, int size);

struct GPUIndex {
    int** indices;
    int* sizes;
    int numCoarseCentroids;
    int numFineCentroids;
};

void checkPointerType(void* ptr);

// 分配并拷贝索引数据到GPU端
void copyIndexToGPU(const std::vector<std::vector<std::vector<idx_t>>>& index, int numCoarseCentroids, int numFineCentroids, GPUIndex* d_index);

// 释放GPU端的索引数据
void freeGPUIndex(GPUIndex& gpuIndex);

class RVQ {
public:
    // 构造函数
    RVQ(int dim, int numCoarseCentroids, int numFineCentroids)
        : dim_(dim), numCoarseCentroid_(numCoarseCentroids), numFineCentroid_(numFineCentroids) {
        coarseCodebook_ = new float[dim * numCoarseCentroids];
        fineCodebook_ = new float[dim * numFineCentroids];
        std::cout << "RVQ index created.\nDimensions = " << dim << ", Coarse Centroids = " << numCoarseCentroids << ", Fine Centroids = " << numFineCentroids << std::endl;
        index_.resize(numCoarseCentroids);
        for (int i = 0; i < numCoarseCentroids; ++i) {
            index_[i].resize(numFineCentroids);
            for (int j = 0; j < numFineCentroids; ++j) {
                index_[i][j].resize(0);
            }
        }

        d_index_ = new GPUIndex;
        d_coarse_codebook_ = nullptr;
        d_fine_codebook_ = nullptr;
    }

    // 析构函数
    ~RVQ() {
        delete[] coarseCodebook_;
        delete[] fineCodebook_;
        // if (d_index_) {
        //     freeGPUIndex(*d_index_);
        //     delete d_index_;
        // }
        // CUDA_CHECK(cudaFree(d_coarse_codebook_));
        // CUDA_CHECK(cudaFree(d_fine_codebook_));
        std::cout << "RVQ object destroyed." << std::endl;
    }

    // 训练粗略量化码本
    void train(std::string base_path, int num_of_points, int num_train_points = -1);

    // 构建精细量化码本
    void build(float* buildVectorData, idx_t numVectors, idx_t base_id);

    // 查询搜索
    void search(float* query, int numQueries, int* cluster);

    void save(const std::string& filename);
    void saveSubgraphData(float* sub_data, std::vector<std::vector<int>> &num_of_subgraph_points, idx_t numBatchPoints, int outIteration);
    void saveSubgraphIndex(int numPoints);
    void load(const std::string& filename);
    void loadCodebook(const std::string& filename);

    std::vector<std::vector<std::vector<idx_t>>> get_index() const;
    int get_numCoarseCentroid() const;
    int get_numFineCentroid() const;
    GPUIndex* get_gpu_index() const;

private:
    int dim_; // 特征向量维度
    float* coarseCodebook_; // 粗码本
    float* d_coarse_codebook_; // GPU上的粗码本
    int numCoarseCentroid_; // 粗码本中心点数量
    float* fineCodebook_; // 细码本
    float* d_fine_codebook_; // GPU上的细码本
    int numFineCentroid_; // 细码本中心点数量
    std::vector<std::vector<std::vector<idx_t>>> index_;
    GPUIndex* d_index_; // GPU的索引数据
};
