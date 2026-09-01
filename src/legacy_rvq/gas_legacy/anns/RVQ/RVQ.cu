/**
 * @author szr
 * @date 2024/5/24
 * @brief two-layer k-means using RVQ method
 *
 * **/

#include <memory>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <limits>
#include <cblas.h>
#include <random>
#include <iostream>
#include <vector>
#include <random>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <string>
// #include </usr/include/mkl/mkl_cblas.h>
// #include </usr/include/mkl/mkl.h>
// #include </usr/include/mkl/mkl_service.h>
#include "RVQ.cuh"
#include "../common.h"
#include "../functions/check.h"
#include "../functions/distance_kernel.cuh"
#include "../functions/selectMin1.cuh"
#include "./select_nearest_cluster.cuh"

namespace {

std::filesystem::path GetEnvDir(const char* name,
                                const std::filesystem::path& fallback) {
    const char* value = std::getenv(name);
    if (value != nullptr && value[0] != '\0') {
        return std::filesystem::path(value);
    }
    return fallback;
}

std::filesystem::path GetRvqOutputDir() {
    return GetEnvDir("ANNS_RVQ_OUTPUT_DIR", std::filesystem::current_path());
}

std::filesystem::path GetSubdataOutputDir() {
    return GetEnvDir("ANNS_SUBDATA_DIR", GetRvqOutputDir() / "subdata");
}

std::string CodebookFileName(int coarse,
                             int fine,
                             idx_t selected_train_vectors,
                             idx_t total_train_vectors) {
    return "Codebook_" + std::to_string(coarse) + "_" + std::to_string(fine) +
           "_" + std::to_string(selected_train_vectors) + "_" +
           std::to_string(total_train_vectors) + ".bin";
}

}  // namespace


void checkPointerType(void* ptr) {
    cudaPointerAttributes attributes;
    cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

    if (err != cudaSuccess) {
        std::cerr << "cudaPointerGetAttributes failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    switch (attributes.type) {
        case cudaMemoryTypeHost:
            std::cout << "Pointer is a host (CPU) pointer." << std::endl;
            break;
        case cudaMemoryTypeDevice:
            std::cout << "Pointer is a device (GPU) pointer." << std::endl;
            break;
        case cudaMemoryTypeManaged:
            std::cout << "Pointer is a managed memory pointer." << std::endl;
            break;
        default:
            std::cout << "Pointer type is unknown." << std::endl;
            break;
    }
}

// 分配并拷贝索引数据到GPU端
void copyIndexToGPU(const std::vector<std::vector<std::vector<idx_t>>>& index, int numCoarseCentroids, int numFineCentroids, GPUIndex* d_index) {

    d_index->numCoarseCentroids = numCoarseCentroids;
    d_index->numFineCentroids = numFineCentroids;

    // 分配指针数组
    int** hostIndices = new int*[numCoarseCentroids * numFineCentroids];
    int* hostSizes = new int[numCoarseCentroids * numFineCentroids];

    // 分配数据并拷贝到GPU
    for (int i = 0; i < numCoarseCentroids; ++i) {
        for (int j = 0; j < numFineCentroids; ++j) {
            int idx = i * numFineCentroids + j;
            hostSizes[idx] = index[i][j].size();
            if (hostSizes[idx] > 0) {
                CUDA_CHECK(cudaMalloc((void**)&hostIndices[idx], hostSizes[idx] * sizeof(idx_t)));
                CUDA_CHECK(cudaMemcpy(hostIndices[idx], index[i][j].data(), hostSizes[idx] * sizeof(idx_t), cudaMemcpyHostToDevice));
            } else {
                hostIndices[idx] = nullptr;
            }
        }
    }

    // 分配GPU端指针
    int** deviceIndices;
    CUDA_CHECK(cudaMalloc((void**)&deviceIndices, numCoarseCentroids * numFineCentroids * sizeof(int*)));
    int* deviceSizes;
    CUDA_CHECK(cudaMalloc((void**)&deviceSizes, numCoarseCentroids * numFineCentroids * sizeof(int)));

    // 拷贝指针数组到GPU
    CUDA_CHECK(cudaMemcpy(deviceIndices, hostIndices, numCoarseCentroids * numFineCentroids * sizeof(int*), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceSizes, hostSizes, numCoarseCentroids * numFineCentroids * sizeof(int), cudaMemcpyHostToDevice));

    // 设置 GPUIndex 的成员
    d_index->indices = deviceIndices;
    d_index->sizes = deviceSizes;

    // 释放临时数组
    delete[] hostIndices;
    delete[] hostSizes;
}

// 释放GPU端的索引数据
void freeGPUIndex(GPUIndex& gpuIndex) {
    if (gpuIndex.indices != nullptr) {
        int totalClusters = gpuIndex.numCoarseCentroids * gpuIndex.numFineCentroids;
        int* hostSizes = new int[totalClusters];

        int** hostIndices = new int*[totalClusters];
        CUDA_CHECK(cudaMemcpy(hostIndices, gpuIndex.indices, totalClusters * sizeof(int*), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaMemcpy(hostSizes, gpuIndex.sizes, totalClusters * sizeof(int), cudaMemcpyDeviceToHost));

        for (int i = 0; i < totalClusters; ++i) {
            if (hostSizes[i] > 0) {
                CUDA_CHECK(cudaFree(hostIndices[i]));
            }
        }

        delete[] hostSizes;

        checkPointerType(gpuIndex.indices);
        checkPointerType(gpuIndex.sizes);
        CUDA_CHECK(cudaFree(gpuIndex.indices));
        CUDA_CHECK(cudaFree(gpuIndex.sizes));
    }
}


// GPU 计算最终clusterId的kernel
__global__ void addKernel(int* d_min_coarse_indices, int* d_min_fine_indices, int* d_enter_cluster, int numFineCentroid, int numQueries) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numQueries) {
        d_enter_cluster[idx] = d_min_coarse_indices[idx] * numFineCentroid + d_min_fine_indices[idx];
    }
}

// // GPU 计算残差的 kernel
// __global__ void computeResiduals(const float* coarse_codebook, const int* min_coarse_indices, float* fine_data, int dim, int num_queries, cublasHandle_t handle) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < num_queries) {
//         int assign_id = min_coarse_indices[idx];
//         const float alpha = -1.0f;
//         cublasSaxpy(handle, dim, &alpha, coarse_codebook + assign_id * dim, 1, fine_data + idx * dim, 1);
//     }
// }

// 手动计算残差的 kernel
__global__ void computeResiduals(const float* coarse_codebook, const int* min_coarse_indices, float* fine_data, int dim, int num_queries) {
    int query_idx = blockIdx.x;
    int dim_idx = threadIdx.x;

    if (query_idx < num_queries && dim_idx < dim) {
        int assign_id = min_coarse_indices[query_idx];
        fine_data[query_idx * dim + dim_idx] -= coarse_codebook[assign_id * dim + dim_idx];
    }
}


std::mt19937 _rnd(time(0));

float kmeans(float* trainData, int numTrainData, int dim, float* codebook, int numCentroids, int* assign) {
    float error = 0.0;

    // 初始化聚类中心为前 numCentroids 个训练数据点
    for (int i = 0; i < numCentroids; ++i) {
        for (int d = 0; d < dim; ++d) {
            codebook[i * dim + d] = trainData[i * dim + d];
        }
    }

    // 迭代更新聚类中心，假设收敛条件为聚类中心不再变化
    bool converged = false;
    while (!converged) {
        converged = true;

        // 对每个训练数据进行分配，并更新聚类中心
        int* counts = new int[numCentroids](); // 用于计算每个聚类的数据点数目
        float* newCentroids = new float[numCentroids * dim](); // 存储新的聚类中心的临时数组

        for (int i = 0; i < numTrainData; ++i) {
            float minDist = std::numeric_limits<float>::max();
            int minIndex = -1;

            // 计算训练数据点与每个聚类中心的距离，找到最近的聚类中心
            for (int j = 0; j < numCentroids; ++j) {
                float dist = 0.0;
                for (int d = 0; d < dim; ++d) {
                    float diff = trainData[i * dim + d] - codebook[j * dim + d];
                    dist += diff * diff;
                }
                if (dist < minDist) {
                    minDist = dist;
                    minIndex = j;
                }
            }

            // 更新分配
            assign[i] = minIndex;

            // 更新聚类中心的临时数组
            for (int d = 0; d < dim; ++d) {
                newCentroids[minIndex * dim + d] += trainData[i * dim + d];
            }
            counts[minIndex]++;

            // 累加误差
            error += minDist;
        }

        // 计算新的聚类中心并检查是否收敛
        for (int i = 0; i < numCentroids; ++i) {
            if (counts[i] > 0) {
                for (int d = 0; d < dim; ++d) {
                    float newCentroid = newCentroids[i * dim + d] / counts[i];
                    if (codebook[i * dim + d] != newCentroid) {
                        converged = false; // 聚类中心有变化，继续迭代
                    }
                    codebook[i * dim + d] = newCentroid;
                }
            }
        }

        // 释放临时数组的内存
        delete[] counts;
        delete[] newCentroids;
    }

    return error;
}

// rand_perm 函数实现
void rand_perm(int* perm, size_t n, int64_t seed) {
    for (size_t i = 0; i < n; i++)
        perm[i] = i;

    std::mt19937 rng(seed);

    for (size_t i = 0; i + 1 < n; i++) {
        std::uniform_int_distribution<size_t> dist(i, n - 1);
        size_t i2 = dist(rng);
        std::swap(perm[i], perm[i2]);
    }
}

// Function to fill an array with random values
void fillWithRandom(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen)*100;
    }
}

// subsample_training_set 函数实现
void subsample_training_set(
    std::string base_path,
    idx_t numVector,
    int dim,
    idx_t numTrainVec,
    float* trainVectors,
    int64_t seed = 1234) {

    // 生成随机排列
    std::vector<int> perm(numTrainVec);
    std::uniform_int_distribution<> dis(0, numVector - 1);
    std::mt19937 rng(seed);
    for (int i = 0; i < numTrainVec; ++i) {
        perm[i] = dis(rng); // 生成并填充随机数
    }
    for (int i = 0; i < numTrainVec; ++i) {
        perm[i] = i; // 生成并填充随机数
    }
    // rand_perm(perm.data(), numVector, seed);


    // 选择前 numTrainVec 个随机向量作为训练数据
    std::ifstream vector_file(base_path, std::ios::binary);
    // for (size_t i = 0; i < numTrainVec; i++) {
    //     unsigned char tmp_data[dim];
    //     vector_file.seekg(size_t(perm[i]) * size_t(dim + 4), std::ios::beg);
    //     vector_file.seekg(4, std::ios::cur);
    //     vector_file.read((char*)(tmp_data), dim);
    //     // printf("Id: %d, %d\n", i, perm[i]);
    //     for(size_t l = 0;l < dim; l++){
    //         trainVectors[i * size_t(dim) + l] = float(tmp_data[l]);
    //     }
    // }

    //deep
    vector_file.seekg(8, std::ios::beg);
    vector_file.read((char*)(trainVectors), size_t(dim) * size_t(numTrainVec) * sizeof(float));
    for(int i = 0; i < dim; i++){
        printf("%f ", trainVectors[i]);
    }
    printf("\n");
    vector_file.close();
    // for (idx_t i = 0; i < numTrainVec; i++) {
    //     std::memcpy(trainVectors + i * dim, dataset + perm[i] * dim, sizeof(float) * dim);
    // }
}

std::vector<int> findMinIndicesCPU(const float* values, int c, int num) {
    std::vector<int> minIndices(num);
    for (int idx = 0; idx < num; ++idx) {
        int startIdx = idx * c;
        int endIdx = (idx + 1) * c;
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
    return minIndices;
}

int rouletteSelection(std::vector<float>& wheel) {
    float total_val = 0;

    for (auto& val : wheel) {
        total_val += val;
    }

    cblas_sscal(wheel.size(), 1.0 / total_val, wheel.data(), 1);
    std::uniform_real_distribution<double> dis(0, 1.0);
    double rd = dis(_rnd);

    for (auto id = 0; id < wheel.size();  ++id) {
        rd -= wheel[id];

        if (rd < 0) {
            return id;
        }
    }

    return wheel.size() - 1;
}

void KmeansppInitCenter(const int total_cnt, const int sample_cnt,
                                  const int dim,
                                  const float* train_dataset, std::vector<int>& sample_ids) {
    std::vector<float> disbest(total_cnt, std::numeric_limits<float>::max()); // 每个点到最近中心的最小距离
    std::vector<float> distmp(total_cnt); // 临时距离存储
    sample_ids.resize(sample_cnt, 0); // 初始化中心索引数组
    sample_ids[0] = _rnd() % total_cnt; // 随机选择第一个中心

    std::vector<float> points_norm(total_cnt, 0); // 每个点的范数

    // 计算每个数据点的范数
    #pragma omp parallel for schedule(dynamic) num_threads(_params.nt)
    for (size_t j = 0; j < total_cnt; j++) {
        points_norm[j] = cblas_sdot(dim, train_dataset + j * dim, 1, train_dataset + j * dim, 1);
    }

    // 选择剩余的初始中心
    for (size_t i = 1; i < sample_cnt; i++) {
        size_t newsel = sample_ids[i - 1]; // 当前选择的中心索引
        const float* last_center = train_dataset + newsel * dim; // 当前选择的中心数据

        #pragma omp parallel for schedule(dynamic) num_threads(_params.nt)
        for (size_t j = 0; j < total_cnt; j++) {
            float temp = points_norm[j] + points_norm[newsel] - 2.0 * cblas_sdot(dim, train_dataset + j * dim, 1,
                         last_center, 1); // 计算距离平方

            if (temp < disbest[j]) {
                disbest[j] = temp; // 更新最近中心的距离
            }
        }

        memcpy(distmp.data(), disbest.data(), total_cnt * sizeof(distmp[0])); // 复制距离数据
        sample_ids[i] = rouletteSelection(distmp); // 轮盘赌选择下一个中心
    }
}

float Kmeans (float* trainData, idx_t numTrainData, int dim, float* codebook, int numCentroids, int* assign) {
    printf("[Info] Start Kmeans training\n");

    // 使用 K-means++ 初始化中心
    std::vector<int> sample_ids;
    KmeansppInitCenter(numTrainData, numCentroids, dim, trainData, sample_ids);

    // 初始化聚类中心为 K-means++ 选择的中心
    for (int i = 0; i < numCentroids; ++i) {
        for (int d = 0; d < dim; ++d) {
            codebook[i * dim + d] = trainData[sample_ids[i] * dim + d];
        }
    }

    float* bestCentroids = new float[numCentroids * dim];

    int iter = 25;
    float* centroids = new float[numCentroids * dim];
    memcpy(centroids, trainData, numCentroids * dim * sizeof(float));
    float* disMatrix = new float[numCentroids * numTrainData];
    std::vector<int> minCentroids;
    std::vector<int> bestMinCentroids(numTrainData, 0);
    std::vector<int> countCentroidPoints(numCentroids, 0);

    float err;
    float minErr = std::numeric_limits<float>::max();

    // loop
    for (int i = 0; i < iter; ++i) {
        printf("Training iteration %d. ", i);
        // distance calculation
        queryToBaseDistance(centroids, numCentroids, trainData, numTrainData, dim, disMatrix);

        // find the nearest centroids of data points
        minCentroids = findMinIndices(disMatrix,  numCentroids, numTrainData);
        // // 验证
        // std::vector<int> cpuResults = findMinIndicesCPU(disMatrix, numCentroids, numTrainData);
        // for (int i = 0; i < numTrainData; ++i) {
        //     if (cpuResults[i] != minCentroids[i]) {
        //         std::cerr << "Mismatch at index " << i << ": CPU result = " << cpuResults[i] << ", GPU result = " << minCentroids[i] << std::endl;
        //     }
        // }

        std::fill(countCentroidPoints.begin(), countCentroidPoints.end(), 0);
        for (int i = 0; i < numTrainData; ++i) {
            countCentroidPoints[minCentroids[i]]++;
        }

        // printf("Points of each centroid: \n");
        // for (int i = 0; i < numCentroids; ++i) {
        //     printf("centroid%d : %d ", i, countCentroidPoints[i]);
        // }
        // printf("\n");

        // printf("Total %d centroids, minCentroids\n", numCentroids);
        // for (int i = 0; i < 20; ++i) {
        //     printf("[%d] = %d ", i, minCentroids[i]);
        // }
        // printf("\n");

        // record the best centroids with min error
        err = 0.0;
        for (int i = 0; i < numTrainData; ++i) {
            err += disMatrix[i*numCentroids + minCentroids[i]];
        }
        // printf("error = %.2f\n", err);
        if (err < minErr) {
            // error < minErr, update minErr, record bestCentroids and bestMinCentroids
            printf("error = %.2f < minErr = %.2f, update minErr\n", err, minErr);
            minErr = err;
            memcpy(bestCentroids, centroids, numCentroids * dim * sizeof(float));
            memcpy(bestMinCentroids.data(), minCentroids.data(), numTrainData * sizeof(int));
        }

        // 用聚类中数据点的平均值更新聚类中心
        // 如果聚类中心里没有数据点，则保留；如果有，则使用数据点平均值代替它
        // printf("[Info] Centroids with no cluster point in it:");
        for (int i = 0; i < numCentroids; ++i) {
            // printf("[%d] = %d ", i, countCentroidPoints[i]);
            if (countCentroidPoints[i] == 0) {
                // printf("centroid[%d] ", i);
            } else {
                memset(centroids + i*dim, 0, dim*sizeof(float));
                // centroids[i] = 0;
            }
        }
        // printf("\n");

        for (int i = 0; i < numTrainData; ++i) {
            for (int j = 0; j < dim; ++j) {
                centroids[minCentroids[i]*dim+j] += (trainData[i*dim+j] / countCentroidPoints[minCentroids[i]]);
            }
        }
    }
    printf("[Info] Finish Kmeans training\n");
    memcpy(codebook, bestCentroids, numCentroids * dim * sizeof(float));
    memcpy(assign, minCentroids.data(), numTrainData * sizeof(int));

    return minErr;
}

std::vector<std::vector<std::vector<idx_t>>> RVQ::get_index() const{
    return index_;
}

int RVQ::get_numCoarseCentroid() const{
    return numCoarseCentroid_;
}

int RVQ::get_numFineCentroid() const{
    return numFineCentroid_;
}

GPUIndex* RVQ::get_gpu_index() const {
    return d_index_;
}

// 训练粗略量化码本
void RVQ::train(std::string base_path,
                idx_t numTrainVectors,
                int numTrainPoints) {
    std::cout << "Training input : " << numTrainVectors << " vectors." << std::endl;
    // for(int i = 0; i<numTrainVectors; i++){
    //     std::cout<<i<<std::endl;
    //     for(int l=0; l<128; l++){
    //         std::cout<<trainVectorData[i*128 + l]<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    // 采样训练点，不超过10w
    idx_t numSelectTrainVec =
        numTrainPoints > 0 ? static_cast<idx_t>(numTrainPoints) : 1000000;
    // 检查输入参数是否有效
    if (numSelectTrainVec > numTrainVectors) {
        std::cout << "Number of select training vectors : " << numTrainVectors << std::endl;
        numSelectTrainVec = numTrainVectors;
    } else {
        std::cout << "Number of select training vectors : " << numSelectTrainVec << std::endl;
    }

    float* selectTrainVectors = new float[numSelectTrainVec * dim_];
    subsample_training_set(base_path, numTrainVectors, dim_, numSelectTrainVec, selectTrainVectors);

    // 迭代的最小误差
    float min_err = std::numeric_limits<float>::max();
    // 每次k-means的训练数据 根据粗细码本迭代更新
    std::unique_ptr<float[]> trainData(new float[numSelectTrainVec * dim_]);
    memcpy(trainData.get(), selectTrainVectors, sizeof(float) * numSelectTrainVec * dim_);
    // 每次kmeans产生的聚类中心
    std::unique_ptr<float[]> coarseCodebook(new float[numCoarseCentroid_ * dim_]);
    std::unique_ptr<int[]> coarseCodebookAssign(new int[numSelectTrainVec]);
    std::unique_ptr<float[]> fineCodebook(new float[numFineCentroid_ * dim_]);
    std::unique_ptr<int[]> fineCodebookAssign(new int[numSelectTrainVec]);

    // 第一层 迭代训练数据 重复调用k-means 参考Tinker
    int iter = 10;
    int niter = 30;
    for (int i = 0; i < iter; ++i) {
        printf("[Info] RVQ training iteration %d\n", i);
        // 使用数据点与二级聚类中心的残差 训练一级聚类中心
        // 第一轮可以理解为二级聚类初始为空
        // Todo: debug Kmeans(), the coarse centroids become the same value after some iterations
        printf("[Info] Training coarse codebook\n");
        float err = Kmeans(trainData.get(), numSelectTrainVec, dim_, coarseCodebook.get(), numCoarseCentroid_, coarseCodebookAssign.get());
        std::cout << "The" << i << " deviation error of coarse clusters is " << err << std::endl;
        std::cout << std::endl;

        // 计算数据点与一级聚类中心的残差 训练二级聚类中心
        memcpy(trainData.get(), selectTrainVectors, sizeof(float) * numSelectTrainVec * dim_);
        for (int i = 0; i < numSelectTrainVec; ++i) {
            int assign_id = coarseCodebookAssign.get()[i];
            cblas_saxpy(dim_, -1.0, coarseCodebook.get() + assign_id * dim_, 1,
                        trainData.get() + i * dim_, 1);
        }

        printf("[Info] Training fine codebook\n");
        err = Kmeans(trainData.get(), numSelectTrainVec, dim_, fineCodebook.get(), numFineCentroid_, fineCodebookAssign.get());
        std::cout << "The " << i << " deviation error of fine clusters is " << err << std::endl;
        std::cout << std::endl;

        // 记录最小误差的训练结果 判定训练是否收敛
        if ((min_err - err) >= 1e-4) { // 参数正在收敛
            memcpy(coarseCodebook_, coarseCodebook.get(),
                   sizeof(float) * numCoarseCentroid_ * dim_);
            memcpy(fineCodebook_, fineCodebook.get(),
                   sizeof(float) * numFineCentroid_ * dim_);
            min_err = err;
        } else { // 开始出现抖动
            std::cout << "current deviation error > min deviation error : " << err << " / " << min_err <<
                      ", params.niter = " << niter << std::endl;

            // niter初值为30，大于80跳出
            if (niter > 80) {
                break;
            }

            niter += 10;
        }

        // 计算数据点与二级聚类中心的残差
        memcpy(trainData.get(), selectTrainVectors, sizeof(float) * numSelectTrainVec * dim_);
        for (idx_t i = 0; i < numSelectTrainVec; ++i) {
            //每次迭代计算T之后的值作为判断标准，所以每次S的值使用最新计算的
            int assign_id = fineCodebookAssign.get()[i];
            cblas_saxpy(dim_, -1.0, fineCodebook.get() + assign_id * dim_, 1,
                        trainData.get() + i * dim_, 1);
        }

    }
    //保存聚类中心
    const std::filesystem::path codebook_path =
        GetRvqOutputDir() /
        CodebookFileName(numCoarseCentroid_,
                         numFineCentroid_,
                         numSelectTrainVec,
                         numTrainVectors);
    std::filesystem::create_directories(codebook_path.parent_path());
    std::ofstream out(codebook_path, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for saving." << std::endl;
        return;
    }

    // 保存维度、聚类中心数量
    out.write(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<char*>(&numCoarseCentroid_), sizeof(numCoarseCentroid_));
    out.write(reinterpret_cast<char*>(&numFineCentroid_), sizeof(numFineCentroid_));

    // 保存粗聚类中心
    out.write(reinterpret_cast<char*>(coarseCodebook_), numCoarseCentroid_ * dim_ * sizeof(float));

    // 保存细聚类中心
    out.write(reinterpret_cast<char*>(fineCodebook_), numFineCentroid_ * dim_ * sizeof(float));
    out.close();

}

__global__ void testKernel(int** d_indices, int* d_sizes, int* output, int* sizes_output, int numCoarseCentroids, int numFineCentroids) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numCoarseCentroids * numFineCentroids) {
        int size = d_sizes[idx];
        sizes_output[idx] = size;
        if (size > 0) {
            output[idx] = d_indices[idx][0]; // 只读取每个cluster中的第一个元素进行测试
        } else {
            output[idx] = -1; // 标记为空的情况
        }
    }
}
void testIndices(GPUIndex* d_index, int numCoarseCentroids, int numFineCentroids) {
    int totalClusters = numCoarseCentroids * numFineCentroids;
    int* h_output = new int[totalClusters];
    int* h_sizes_output = new int[totalClusters];
    int* d_output;
    int* d_sizes_output;
    cudaMalloc(&d_output, totalClusters * sizeof(int));
    cudaMalloc(&d_sizes_output, totalClusters * sizeof(int));

    // 启动核函数
    int blockSize = 256;
    int numBlocks = (totalClusters + blockSize - 1) / blockSize;
    testKernel<<<numBlocks, blockSize>>>(d_index->indices, d_index->sizes, d_output, d_sizes_output, numCoarseCentroids, numFineCentroids);
    cudaDeviceSynchronize();

    // 将结果从设备端拷贝到主机端
    cudaMemcpy(h_output, d_output, totalClusters * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sizes_output, d_sizes_output, totalClusters * sizeof(int), cudaMemcpyDeviceToHost);

    // 打印结果
    for (int i = 0; i < totalClusters; ++i) {
        std::cout << "Cluster " << i << " size: " << h_sizes_output[i];
        if (h_output[i] != -1) {
            std::cout << ", first element: " << h_output[i] << std::endl;
        } else {
            std::cout << ", is empty." << std::endl;
        }
    }

    // 释放内存
    delete[] h_output;
    delete[] h_sizes_output;
    cudaFree(d_output);
    cudaFree(d_sizes_output);
}

__global__ void cal_fine_data(float* d_fine_data, float* d_coarse_codebook, int* d_min_coarse_indices, int dim_){

    for (int i = threadIdx.x; i < dim_; i += blockDim.x) {
        d_fine_data[blockIdx.x*dim_ + i] -= d_coarse_codebook[d_min_coarse_indices[blockIdx.x]*dim_ + i];
    }

}

// 构建反向索引
void RVQ::build(float* d_buildVectorData, num_t num, idx_t base_id) {

    // Todo: 距离计算没有分块，可能放不下
    // Todo: fuse (distance + 1-selection) 放入一个kernel?
    int k = 1;

    const int numVectors = 100000;
    const int iteration = static_cast<int>((num + numVectors - 1) / numVectors);
    int* d_result;
    cudaMalloc((void**)&d_result, numVectors * k * sizeof(int));
    int* h_result = new int[numVectors * k];
    // std::ostringstream filename1;
    // std::ofstream file1(filename1.str(), std::ios::binary | std::ios::app);

    // std::ostringstream filename2;
    // std::ofstream file2(filename2.str(), std::ios::binary | std::ios::app);
    for(int iter = 0; iter < iteration; iter++){
        //printf("%d\n",iter);
        int tmpNumVectors = numVectors;
        if(iter == iteration - 1){
            tmpNumVectors = num - iter * numVectors;
        }
        constexpr int WARP_SIZE = 32;
        constexpr int NumWarpQ = 32;
        constexpr int NumThreadQ = 1;
        selectNearestCluster<int, float, WARP_SIZE, NumWarpQ, NumThreadQ><<<tmpNumVectors, 32, min(numCoarseCentroid_, NumWarpQ) * sizeof(KernelPair<float, int>)>>>(
                            d_buildVectorData + iter * numVectors * dim_, d_coarse_codebook_, d_fine_codebook_, d_result, tmpNumVectors,
                            dim_, k, numCoarseCentroid_, numFineCentroid_);
        // // 计算与粗聚类中心距离
        // float* d_dis_matrix_coarse;
        // cudaMalloc((void**)&d_dis_matrix_coarse, tmpNumVectors * numCoarseCentroid_ * sizeof(float));
        // deviceQueryToBaseDistance(d_coarse_codebook_, numCoarseCentroid_, d_buildVectorData + iter * numVectors * dim_, tmpNumVectors, dim_, d_dis_matrix_coarse, 100000);

        // // 得到最近的粗聚类中心
        // int* d_min_coarse_indices;
        // cudaMalloc((void**)&d_min_coarse_indices, tmpNumVectors * sizeof(int));
        // deviceFindMinIndices(d_dis_matrix_coarse, numCoarseCentroid_, tmpNumVectors, d_min_coarse_indices);
        // // std::vector<int> minCoarseIndices = findMinIndices(disMatrixCoarse, numCoarseCentroid_, numVectors);

        // // 计算残差
        // float* d_fine_data;
        // CUDA_CHECK(cudaMalloc((void**)&d_fine_data, tmpNumVectors * dim_ * sizeof(float)));
        // CUDA_CHECK(cudaMemcpy(d_fine_data, d_buildVectorData + iter * numVectors * dim_, tmpNumVectors * dim_ * sizeof(float), cudaMemcpyDeviceToDevice));
        // CUDA_SYNC_CHECK();
        // cal_fine_data<<<tmpNumVectors,128>>>(d_fine_data, d_coarse_codebook_, d_min_coarse_indices, dim_);

        // // 得到最近的细聚类中心
        // float* d_dis_matrix_fine;
        // cudaMalloc((void**)&d_dis_matrix_fine, tmpNumVectors * numFineCentroid_ * sizeof(float));

        // deviceQueryToBaseDistance(d_fine_codebook_, numFineCentroid_, d_fine_data,
        //                     tmpNumVectors, dim_, d_dis_matrix_fine, 100000);
        // int* d_min_fine_indices;
        // cudaMalloc((void**)&d_min_fine_indices, tmpNumVectors * sizeof(int));
        // deviceFindMinIndices(d_dis_matrix_fine, numFineCentroid_, tmpNumVectors, d_min_fine_indices);

        // 加入反向列表
        // int* minCoarseIndices = new int[tmpNumVectors];
        // cudaMemcpy(minCoarseIndices, d_min_coarse_indices, tmpNumVectors * sizeof(int), cudaMemcpyDeviceToHost);
        // int* minFineIndices = new int[tmpNumVectors];
        // cudaMemcpy(minFineIndices, d_min_fine_indices, tmpNumVectors * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_result, d_result, tmpNumVectors * k * sizeof(int), cudaMemcpyDeviceToHost);
        for(idx_t i = 0; i < tmpNumVectors; ++i){
            for(idx_t l = 0; l < k; ++l){
                const int cluster_id = h_result[i * k + l];
                const int coarse_id = cluster_id / numFineCentroid_;
                const int fine_id = cluster_id % numFineCentroid_;
                if (coarse_id < 0 || coarse_id >= numCoarseCentroid_ ||
                    fine_id < 0 || fine_id >= numFineCentroid_) {
                    continue;
                }
                index_[coarse_id][fine_id].push_back(
                    static_cast<idx_t>(i + static_cast<idx_t>(numVectors) * iter + base_id));
            }
        }
        // cudaFree(d_dis_matrix_coarse);
        // cudaFree(d_min_coarse_indices);
        // cudaFree(d_fine_data);
        // cudaFree(d_dis_matrix_fine);
        // cudaFree(d_min_fine_indices);
    }
    delete[] h_result;
    cudaFree(d_result);

}

// 查询搜索
void RVQ::search(float* d_query, int numQueries, int* d_enter_cluster) {
    std::cout << "Searching with " << numQueries << " queries." << std::endl;

    Timer queryT;
    //queryT.Start();
    // 计算与粗聚类中心距离
    //Todo: modify queryToBaseDistance() input to GPU query input
    //Todo: use GPU memory disMatrix
    float* d_dis_matrix_coarse;
    cudaMalloc((void**)&d_dis_matrix_coarse, numQueries * numCoarseCentroid_ * sizeof(float));
    deviceQueryToBaseDistance(d_coarse_codebook_, numCoarseCentroid_, d_query, numQueries, dim_, d_dis_matrix_coarse, 100000);
    //queryT.Stop();
    //std::cout<<"[RVQ] distance to coarse centroids time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
    CUDA_SYNC_CHECK();

    //queryT.Start();
    int* d_min_coarse_indices;
    cudaMalloc((void**)&d_min_coarse_indices, numQueries * sizeof(int));
    // 得到最近的粗聚类中心
    deviceFindMinIndices(d_dis_matrix_coarse, numCoarseCentroid_, numQueries, d_min_coarse_indices);
    //queryT.Stop();
    //std::cout<<"[RVQ] coarse centroids findMinIndices time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
    CUDA_SYNC_CHECK();

    // 分配残差计算所需的内存
    float* d_fine_data;
    CUDA_CHECK(cudaMalloc((void**)&d_fine_data, numQueries * dim_ * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_fine_data, d_query, numQueries * dim_ * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_SYNC_CHECK();

    // 计算残差
    // Todo: change cblas_saxpy to cublasSaxpy
    //queryT.Start();

    // // 定义 kernel 线程配置
    // int block_size = 256;
    // int num_blocks = (numQueries + block_size - 1) / block_size;

    // cublasHandle_t cublas_handle;
    // cublasCreate(&cublas_handle);
    // // 启动 kernel 计算残差
    // computeResiduals<<<num_blocks, block_size>>>(d_coarse_codebook_, d_min_coarse_indices, d_fine_data, dim_, numQueries, cublas_handle);
    // cublasDestroy(cublas_handle);

    dim3 block_size(dim_); // 每个block的线程数等于dim
    dim3 num_blocks(numQueries); // block的数量等于查询数量

    // 启动 kernel 计算残差
    computeResiduals<<<num_blocks, block_size>>>(d_coarse_codebook_, d_min_coarse_indices, d_fine_data, dim_, numQueries);
    CUDA_SYNC_CHECK();

    //queryT.Stop();
    //std::cout<<"[RVQ] calculate residual time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;


    //queryT.Start();
    // 得到最近的细聚类中心
    float* d_dis_matrix_fine;
    cudaMalloc((void**)&d_dis_matrix_fine, numQueries * numFineCentroid_ * sizeof(float));

    deviceQueryToBaseDistance(d_fine_codebook_, numFineCentroid_, d_fine_data,
                         numQueries, dim_, d_dis_matrix_fine, 100000);
    //queryT.Stop();
    //std::cout<<"[RVQ] distance to fine centroids time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;

    // 得到最近的细聚类中心
    //queryT.Start();
    // int* d_min_fine_indices;
    // cudaMalloc((void**)&d_min_fine_indices, numQueries * sizeof(int));
    deviceFindMinIndices(d_min_coarse_indices, d_dis_matrix_fine, numFineCentroid_, numQueries, d_enter_cluster, d_index_->sizes);
    //queryT.Stop();
    //std::cout<<"[RVQ] fine centroids findMinIndices time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;


    //Todo: 得到minCoarseIndices和minFineIndices之后，需要进行什么操作？返回index？
    //queryT.Start();
    // addKernel进行加和
    // int blockSize = 256;
    // int numBlocks = (numQueries + blockSize - 1) / blockSize;
    // addKernel<<<numBlocks, blockSize>>>(d_min_coarse_indices, d_min_fine_indices, d_enter_cluster, numFineCentroid_, numQueries);
    // CUDA_SYNC_CHECK();

    // for(idx_t i = 0; i < numQueries; ++i){
    //     enter_cluster.push_back(minCoarseIndices[i] * numFineCentroid_ + minFineIndices[i]);
    //     // if (i < 100){
    //     //     printf("Query %d : coarseId = %d, fineId = %d\n", i, minCoarseIndices[i], minFineIndices[i]);
    //     // }
    // }
    //queryT.Stop();
    //std::cout<<"[RVQ] init result vector time: "<<queryT.DurationInMilliseconds()<<" ms"<<std::endl;
}

void RVQ::save(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "Failed to open file for saving." << std::endl;
        return;
    }

    // 保存维度、聚类中心数量
    out.write(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    out.write(reinterpret_cast<char*>(&numCoarseCentroid_), sizeof(numCoarseCentroid_));
    out.write(reinterpret_cast<char*>(&numFineCentroid_), sizeof(numFineCentroid_));

    // 保存粗聚类中心
    out.write(reinterpret_cast<char*>(coarseCodebook_), numCoarseCentroid_ * dim_ * sizeof(float));

    // 保存细聚类中心
    out.write(reinterpret_cast<char*>(fineCodebook_), numFineCentroid_ * dim_ * sizeof(float));

    // 保存索引
    size_t outerSize = index_.size();
    out.write(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
    int Coarse_id = 0;
    int Fine_id = 0;
    for (const auto& inner : index_) {
        size_t innerSize = inner.size();
        size_t Coarse_size = 0;
        out.write(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
        for (const auto& innerInner : inner) {
            size_t innerInnerSize = innerInner.size();
            Coarse_size += innerInnerSize;
            printf("Fine cluster %d size:%lu\n", Fine_id, innerInnerSize);
            Fine_id++;
            out.write(reinterpret_cast<char*>(&innerInnerSize), sizeof(innerInnerSize));
            out.write(reinterpret_cast<char*>(const_cast<idx_t*>(innerInner.data())), innerInnerSize * sizeof(idx_t));
        }
        printf("Coarse cluster %d size:%lu\n", Coarse_id, Coarse_size);
        Coarse_id++;
    }

    out.close();
}

void RVQ::saveSubgraphData(float* sub_data, std::vector<std::vector<int>> &num_of_subgraph_points, idx_t numBatchPoints, int outIteration){
    int offset = numBatchPoints * outIteration;
    for(int i = 0; i < index_.size(); i++){
        for(int l = 0; l < index_[i].size(); l++){
            int end = index_[i][l].size();
            if(end == num_of_subgraph_points[i][l]) continue;
            std::ostringstream filename;
            filename << "subData" << std::setw(4) << std::setfill('0')
                     << i * numFineCentroid_ + l << ".bin";
            const std::filesystem::path subdata_path =
                GetSubdataOutputDir() / filename.str();
            std::filesystem::create_directories(subdata_path.parent_path());
            std::ofstream file(subdata_path, std::ios::binary | std::ios::app);
            if(file.tellp() == 0){
                file.write(reinterpret_cast<char*>(&dim_), sizeof(int));
            }
            for(int start = num_of_subgraph_points[i][l]; start < end; start++){
                int id = index_[i][l][start] - offset;
                // file.write(reinterpret_cast<char*>(&dim_), sizeof(int));
                file.write(reinterpret_cast<char*>(sub_data + id * dim_), sizeof(float) * dim_);
            }
            num_of_subgraph_points[i][l] = index_[i][l].size();
            file.close();
        }
    }
}

void RVQ::saveSubgraphIndex(int numPoints){
    printf("Save subgraph index\n");
    int num_of_cluster = numCoarseCentroid_ * numFineCentroid_;
    int* preFixSizeSubgraph = new int[num_of_cluster + 1]();

    const std::filesystem::path subdata_dir = GetSubdataOutputDir();
    std::filesystem::create_directories(subdata_dir);
    std::ofstream file2(subdata_dir / "new_index_of_data.bin", std::ios::binary);
    // file2.write(reinterpret_cast<char*>(&numPoints), sizeof(int));

    for(int i = 0; i < index_.size(); i++){
        for(int l = 0; l < index_[i].size(); l++){
            file2.write(reinterpret_cast<char*>(index_[i][l].data()), sizeof(int) * index_[i][l].size());
            preFixSizeSubgraph[i * numFineCentroid_ + l + 1] = preFixSizeSubgraph[i * numFineCentroid_ + l] + index_[i][l].size();
        }
    }
    file2.close();
    printf("There are %d points in total\n",preFixSizeSubgraph[num_of_cluster]);
    std::ofstream file1(subdata_dir / "pre_fix_of_subgraph_size.bin", std::ios::binary);
    file1.write(reinterpret_cast<char*>(&(num_of_cluster)), sizeof(int));
    file1.write(reinterpret_cast<char*>(preFixSizeSubgraph), sizeof(int) * (num_of_cluster + 1));
    file1.close();
}

void RVQ::load(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading." << std::endl;
        return;
    }

    // 加载维度、聚类中心数量
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&numCoarseCentroid_), sizeof(numCoarseCentroid_));
    in.read(reinterpret_cast<char*>(&numFineCentroid_), sizeof(numFineCentroid_));

    // 分配内存
    delete[] coarseCodebook_;
    delete[] fineCodebook_;
    coarseCodebook_ = new float[numCoarseCentroid_ * dim_];
    fineCodebook_ = new float[numFineCentroid_ * dim_];

    // 加载粗聚类中心
    in.read(reinterpret_cast<char*>(coarseCodebook_), numCoarseCentroid_ * dim_ * sizeof(float));

    // 加载细聚类中心
    in.read(reinterpret_cast<char*>(fineCodebook_), numFineCentroid_ * dim_ * sizeof(float));

    // 加载索引
    int max_size = 0;
    int zero_cluster = 0;
    size_t outerSize;
    in.read(reinterpret_cast<char*>(&outerSize), sizeof(outerSize));
    size_t* points_in_coarseCodebook = new size_t[numCoarseCentroid_]();
    int i = 0;
    index_.resize(outerSize);
    for (auto& inner : index_) {
        size_t innerSize;
        in.read(reinterpret_cast<char*>(&innerSize), sizeof(innerSize));
        inner.resize(innerSize);
        for (auto& innerInner : inner) {
            size_t innerInnerSize;
            in.read(reinterpret_cast<char*>(&innerInnerSize), sizeof(innerInnerSize));
            if(innerInnerSize == 0) zero_cluster++;
            if(innerInnerSize > max_size) max_size = innerInnerSize;
            points_in_coarseCodebook[i]+=innerInnerSize;
            innerInner.resize(innerInnerSize);
            in.read(reinterpret_cast<char*>(innerInner.data()), innerInnerSize * sizeof(idx_t));
        }
        i++;
    }
    // printf("Points in coarseCodebook:");
    // for(i = 0; i < numCoarseCentroid_; i++){
    //     std::cout<<points_in_coarseCodebook[i]<<" ";
    // }
    printf("\n");
    printf("Max cluster size:%d\n",max_size);
    printf("Num of zero cluster:%d\n",zero_cluster);
    delete[] points_in_coarseCodebook;
    in.close();

    // index复制到device
    copyIndexToGPU(index_, numCoarseCentroid_, numFineCentroid_, d_index_);
    // 粗码本和细码本复制到device
    cudaMalloc(&d_coarse_codebook_, numCoarseCentroid_ * dim_ * sizeof(float));
    cudaMemcpy(d_coarse_codebook_, coarseCodebook_, numCoarseCentroid_ * dim_ * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fine_codebook_, numFineCentroid_ * dim_ * sizeof(float));
    cudaMemcpy(d_fine_codebook_, fineCodebook_, numFineCentroid_ * dim_ * sizeof(float), cudaMemcpyHostToDevice);
}

void RVQ::loadCodebook(const std::string& filename){
    //读取聚类中心
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open file for loading." << std::endl;
        return;
    }

    // 加载维度、聚类中心数量
    in.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    in.read(reinterpret_cast<char*>(&numCoarseCentroid_), sizeof(numCoarseCentroid_));
    in.read(reinterpret_cast<char*>(&numFineCentroid_), sizeof(numFineCentroid_));

    // 分配内存
    delete[] coarseCodebook_;
    delete[] fineCodebook_;
    coarseCodebook_ = new float[numCoarseCentroid_ * dim_];
    fineCodebook_ = new float[numFineCentroid_ * dim_];

    // 加载粗聚类中心
    in.read(reinterpret_cast<char*>(coarseCodebook_), numCoarseCentroid_ * dim_ * sizeof(float));

    // 加载细聚类中心
    in.read(reinterpret_cast<char*>(fineCodebook_), numFineCentroid_ * dim_ * sizeof(float));
    in.close();

    // 拷贝粗码本和细码本到GPU
    cudaMalloc(&d_coarse_codebook_, numCoarseCentroid_ * dim_ * sizeof(float));
    cudaMemcpy(d_coarse_codebook_, coarseCodebook_, numCoarseCentroid_ * dim_ * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&d_fine_codebook_, numFineCentroid_ * dim_ * sizeof(float));
    cudaMemcpy(d_fine_codebook_, fineCodebook_, numFineCentroid_ * dim_ * sizeof(float), cudaMemcpyHostToDevice);
}

// int main() {
//     // Define parameters
//     int dim = 128; // Dimension of feature vectors
//     int numCoarseCentroids = 10; // Number of coarse centroids
//     int numFineCentroids = 10; // Number of fine centroids
//     int numTrainVectors = 1000; // Number of training vectors
//     int numQueries = 100; // Number of query vectors

//     // Generate random training data and query data
//     float* trainData = new float[numTrainVectors * dim];
//     float* queryData = new float[numQueries * dim];
//     fillWithRandom(trainData, numTrainVectors * dim);
//     fillWithRandom(queryData, numQueries * dim);

//     float *d_queries;
//     cudaMalloc((void **)&d_queries, sizeof(float) * numQueries * dim);
//     cudaMemcpy(d_queries, queryData, sizeof(float) * numQueries * dim, cudaMemcpyHostToDevice);

//     // Create RVQ object
//     RVQ rvq(dim, numCoarseCentroids, numFineCentroids);

//     // Train RVQ
//     rvq.train(trainData, numTrainVectors);

//     // Build reverse index
//     rvq.build(trainData, numTrainVectors);

//     // // Search using queries
//     int* results;
//     cudaMalloc((void**)&results, numQueries * sizeof(int));
//     rvq.search(d_queries, numQueries, results);
//     cudaDeviceSynchronize();

//     // // Copy results from device to host
//     int* h_results = new int[numQueries];
//     cudaMemcpy(h_results, results, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

//     // // Display results
//     std::cout << "Search results: " << std::endl;
//     for (int i = 0; i < numQueries; ++i) {
//         std::cout << "Query " << i << ": Cluster " << h_results[i] << std::endl;
//     }

//     // // Get index and print statistics
//     // auto index = rvq.get_index();
//     // auto d_index = rvq.get_gpu_index();
//     // for (int i = 0; i < numCoarseCentroids; ++i) {
//     //     for (int j = 0; j < numFineCentroids; ++j) {
//     //         std::cout << "Coarse centroid " << i << ", Fine centroid " << j
//     //                   << " has " << index[i][j].size() << " points." << std::endl;
//     //     }
//     // }

//     rvq.save("rvq_model.bin");

//     RVQ rvq_loaded(dim, numCoarseCentroids, numFineCentroids);
//     rvq_loaded.load("rvq_model.bin");
//     cudaMalloc((void **)&d_queries, sizeof(float) * numQueries * dim);
//     cudaMemcpy(d_queries, queryData, sizeof(float) * numQueries * dim, cudaMemcpyHostToDevice);
//     rvq_loaded.search(d_queries, numQueries, results);

//     // // Copy results from device to host
//     // cudaMemcpy(h_results, results, numQueries * sizeof(int), cudaMemcpyDeviceToHost);

//     // // Display results
//     std::cout << "Search results: " << std::endl;
//     for (int i = 0; i < numQueries; ++i) {
//         std::cout << "Query " << i << ": Cluster " << h_results[i] << std::endl;
//     }

//     // Clean up
//     // delete[] trainData;
//     // delete[] queryData;

//     return 0;
// }
