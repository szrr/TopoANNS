#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "RVQ/RVQ.cuh"

namespace {

std::string ExpectedCodebookName(int coarse,
                                 int fine,
                                 int num_vectors,
                                 int train_points) {
    const int selected =
        train_points > 0 ? std::min(train_points, num_vectors) : std::min(1000000, num_vectors);
    return "Codebook_" + std::to_string(coarse) + "_" + std::to_string(fine) +
           "_" + std::to_string(selected) + "_" +
           std::to_string(num_vectors) + ".bin";
}

void PrepareOutputEnv(const std::filesystem::path& output_model_path) {
    const std::filesystem::path output_dir = output_model_path.parent_path();
    const std::filesystem::path subdata_dir = output_dir / "subdata";
    std::filesystem::create_directories(output_dir);
    std::filesystem::create_directories(subdata_dir);
    setenv("ANNS_OUTPUT_DIR", output_dir.c_str(), 1);
    setenv("ANNS_RVQ_OUTPUT_DIR", output_dir.c_str(), 1);
    setenv("ANNS_SUBDATA_DIR", subdata_dir.c_str(), 1);
}

std::vector<float> LoadRawFloatVectors(const std::filesystem::path& path,
                                       int dim,
                                       int count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open raw float file: " + path.string());
    }
    std::vector<float> vectors(static_cast<std::size_t>(dim) * count, 0.0f);
    in.read(reinterpret_cast<char*>(vectors.data()),
            static_cast<std::streamsize>(vectors.size() * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(vectors.size() * sizeof(float))) {
        throw std::runtime_error("Short read from raw float file: " + path.string());
    }
    return vectors;
}

std::vector<float> LoadFloatXBin(const std::filesystem::path& path,
                                 int dim,
                                 int count) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open float xbin file: " + path.string());
    }
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!in.good() || cols != static_cast<std::uint32_t>(dim) ||
        rows < static_cast<std::uint32_t>(count)) {
        throw std::runtime_error("Invalid float xbin header: " + path.string());
    }

    std::vector<float> vectors(static_cast<std::size_t>(dim) * count, 0.0f);
    in.read(reinterpret_cast<char*>(vectors.data()),
            static_cast<std::streamsize>(vectors.size() * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(vectors.size() * sizeof(float))) {
        throw std::runtime_error("Short read from float xbin file: " + path.string());
    }
    return vectors;
}

void TrainExport(const std::filesystem::path& raw_base_path,
                 int dim,
                 int num_vectors,
                 int train_points,
                 int coarse,
                 int fine,
                 int batch_size,
                 const std::filesystem::path& output_model_path) {
    PrepareOutputEnv(output_model_path);

    RVQ rvq(dim, coarse, fine);
    rvq.train(raw_base_path.string(), num_vectors, train_points);
    const std::filesystem::path codebook_path =
        output_model_path.parent_path() /
        ExpectedCodebookName(coarse, fine, num_vectors, train_points);
    rvq.loadCodebook(codebook_path.string());

    std::ifstream in(raw_base_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open raw base vectors: " +
                                 raw_base_path.string());
    }
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!in.good() || cols != static_cast<std::uint32_t>(dim) ||
        rows < static_cast<std::uint32_t>(num_vectors)) {
        throw std::runtime_error("Invalid float xbin header: " + raw_base_path.string());
    }

    std::vector<float> host_batch(static_cast<std::size_t>(batch_size) * dim, 0.0f);
    int processed = 0;
    while (processed < num_vectors) {
        const int current = std::min(batch_size, num_vectors - processed);
        in.read(reinterpret_cast<char*>(host_batch.data()),
                static_cast<std::streamsize>(static_cast<std::size_t>(current) * dim *
                                             sizeof(float)));
        if (in.gcount() != static_cast<std::streamsize>(
                               static_cast<std::size_t>(current) * dim * sizeof(float))) {
            throw std::runtime_error("Short read while building RVQ index.");
        }

        float* device_batch = nullptr;
        cudaError_t status = cudaMalloc(&device_batch,
                                        static_cast<std::size_t>(current) * dim *
                                            sizeof(float));
        if (status != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed during RVQ build.");
        }
        status = cudaMemcpy(device_batch,
                            host_batch.data(),
                            static_cast<std::size_t>(current) * dim * sizeof(float),
                            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            cudaFree(device_batch);
            throw std::runtime_error("cudaMemcpy failed during RVQ build.");
        }
        rvq.build(device_batch, current, processed);
        cudaFree(device_batch);
        processed += current;
        std::cout << "[topoanns_rvq] built " << processed << " / " << num_vectors
                  << " vectors" << std::endl;
    }

    rvq.save(output_model_path.string());
    std::cout << "[topoanns_rvq] codebook: " << codebook_path << std::endl;
    std::cout << "[topoanns_rvq] model: " << output_model_path << std::endl;
}

void BuildExport(const std::filesystem::path& raw_base_path,
                 int dim,
                 int num_vectors,
                 int coarse,
                 int fine,
                 int batch_size,
                 const std::filesystem::path& codebook_path,
                 const std::filesystem::path& output_model_path) {
    PrepareOutputEnv(output_model_path);

    RVQ rvq(dim, coarse, fine);
    rvq.loadCodebook(codebook_path.string());

    std::ifstream in(raw_base_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open raw base vectors: " +
                                 raw_base_path.string());
    }
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!in.good() || cols != static_cast<std::uint32_t>(dim) ||
        rows < static_cast<std::uint32_t>(num_vectors)) {
        throw std::runtime_error("Invalid float xbin header: " + raw_base_path.string());
    }

    std::vector<float> host_batch(static_cast<std::size_t>(batch_size) * dim, 0.0f);
    int processed = 0;
    while (processed < num_vectors) {
        const int current = std::min(batch_size, num_vectors - processed);
        in.read(reinterpret_cast<char*>(host_batch.data()),
                static_cast<std::streamsize>(static_cast<std::size_t>(current) * dim *
                                             sizeof(float)));
        if (in.gcount() != static_cast<std::streamsize>(
                               static_cast<std::size_t>(current) * dim * sizeof(float))) {
            throw std::runtime_error("Short read while building RVQ index.");
        }

        float* device_batch = nullptr;
        cudaError_t status = cudaMalloc(&device_batch,
                                        static_cast<std::size_t>(current) * dim *
                                            sizeof(float));
        if (status != cudaSuccess) {
            throw std::runtime_error("cudaMalloc failed during RVQ build.");
        }
        status = cudaMemcpy(device_batch,
                            host_batch.data(),
                            static_cast<std::size_t>(current) * dim * sizeof(float),
                            cudaMemcpyHostToDevice);
        if (status != cudaSuccess) {
            cudaFree(device_batch);
            throw std::runtime_error("cudaMemcpy failed during RVQ build.");
        }

        rvq.build(device_batch, current, processed);
        cudaFree(device_batch);
        processed += current;
        std::cout << "[topoanns_rvq] built " << processed << " / " << num_vectors
                  << " vectors" << std::endl;
    }

    rvq.save(output_model_path.string());
    std::cout << "[topoanns_rvq] codebook: " << codebook_path << std::endl;
    std::cout << "[topoanns_rvq] model: " << output_model_path << std::endl;
}

void Verify(const std::filesystem::path& model_path,
            const std::filesystem::path& query_bin_path,
            int dim,
            int num_queries,
            int coarse,
            int fine) {
    RVQ rvq(dim, coarse, fine);
    rvq.load(model_path.string());

    std::vector<float> queries = LoadFloatXBin(query_bin_path, dim, num_queries);
    float* device_queries = nullptr;
    int* device_clusters = nullptr;
    cudaError_t status =
        cudaMalloc(&device_queries, static_cast<std::size_t>(num_queries) * dim * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for RVQ verify queries.");
    }
    status = cudaMalloc(&device_clusters,
                        static_cast<std::size_t>(num_queries) * sizeof(int));
    if (status != cudaSuccess) {
        cudaFree(device_queries);
        throw std::runtime_error("cudaMalloc failed for RVQ verify clusters.");
    }
    status = cudaMemcpy(device_queries,
                        queries.data(),
                        static_cast<std::size_t>(num_queries) * dim * sizeof(float),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(device_queries);
        cudaFree(device_clusters);
        throw std::runtime_error("cudaMemcpy failed for RVQ verify.");
    }

    rvq.search(device_queries, num_queries, device_clusters);
    cudaDeviceSynchronize();

    std::vector<int> clusters(num_queries, -1);
    status = cudaMemcpy(clusters.data(),
                        device_clusters,
                        static_cast<std::size_t>(num_queries) * sizeof(int),
                        cudaMemcpyDeviceToHost);
    cudaFree(device_queries);
    cudaFree(device_clusters);
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMemcpy failed when reading RVQ verify results.");
    }

    std::cout << "[topoanns_rvq] verify clusters:";
    for (int i = 0; i < std::min(num_queries, 8); ++i) {
        std::cout << " " << clusters[i];
    }
    std::cout << std::endl;
}

void BenchmarkXBin(const std::filesystem::path& model_path,
                   const std::filesystem::path& query_bin_path,
                   int dim,
                   int num_queries,
                   int coarse,
                   int fine,
                   int warmup_iters,
                   int measure_iters) {
    if (warmup_iters < 0 || measure_iters <= 0) {
        throw std::runtime_error("warmup_iters must be >= 0 and measure_iters must be > 0.");
    }

    RVQ rvq(dim, coarse, fine);
    rvq.load(model_path.string());
    std::vector<float> queries = LoadFloatXBin(query_bin_path, dim, num_queries);
    float* device_queries = nullptr;
    int* device_clusters = nullptr;
    cudaError_t status =
        cudaMalloc(&device_queries, static_cast<std::size_t>(num_queries) * dim * sizeof(float));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaMalloc failed for RVQ benchmark queries.");
    }
    status = cudaMalloc(&device_clusters,
                        static_cast<std::size_t>(num_queries) * sizeof(int));
    if (status != cudaSuccess) {
        cudaFree(device_queries);
        throw std::runtime_error("cudaMalloc failed for RVQ benchmark clusters.");
    }
    status = cudaMemcpy(device_queries,
                        queries.data(),
                        static_cast<std::size_t>(num_queries) * dim * sizeof(float),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        cudaFree(device_queries);
        cudaFree(device_clusters);
        throw std::runtime_error("cudaMemcpy failed for RVQ benchmark.");
    }
    cudaDeviceSynchronize();

    for (int iter = 0; iter < warmup_iters; ++iter) {
        rvq.search(device_queries, num_queries, device_clusters);
        cudaDeviceSynchronize();
    }

    double first_host_ms = 0.0;
    double first_kernel_ms = 0.0;
    double total_host_ms = 0.0;
    double total_kernel_ms = 0.0;

    for (int iter = 0; iter < measure_iters; ++iter) {
        cudaEvent_t kernel_begin = nullptr;
        cudaEvent_t kernel_end = nullptr;
        cudaEventCreate(&kernel_begin);
        cudaEventCreate(&kernel_end);

        const auto host_begin = std::chrono::steady_clock::now();
        cudaEventRecord(kernel_begin);
        rvq.search(device_queries, num_queries, device_clusters);
        cudaEventRecord(kernel_end);
        cudaEventSynchronize(kernel_end);
        const auto host_end = std::chrono::steady_clock::now();

        float kernel_ms = 0.0f;
        cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end);
        cudaEventDestroy(kernel_begin);
        cudaEventDestroy(kernel_end);

        const double host_ms =
            std::chrono::duration<double, std::milli>(host_end - host_begin).count();
        if (iter == 0) {
            first_host_ms = host_ms;
            first_kernel_ms = static_cast<double>(kernel_ms);
        }
        total_host_ms += host_ms;
        total_kernel_ms += static_cast<double>(kernel_ms);
    }

    cudaFree(device_queries);
    cudaFree(device_clusters);

    std::cout << "[topoanns_rvq] bench num_queries=" << num_queries
              << " warmup_iters=" << warmup_iters
              << " measure_iters=" << measure_iters << std::endl;
    std::cout << "[topoanns_rvq] first_host_ms=" << first_host_ms
              << " first_kernel_ms=" << first_kernel_ms << std::endl;
    std::cout << "[topoanns_rvq] avg_host_ms=" << (total_host_ms / measure_iters)
              << " avg_kernel_ms=" << (total_kernel_ms / measure_iters) << std::endl;
}

[[noreturn]] void Usage() {
    std::cerr
        << "Usage:\n"
        << "  topoanns_rvq_tool train_export <base_xbin> <dim> <num_vectors> <train_points> <coarse> <fine> <batch_size> <output_model>\n"
        << "  topoanns_rvq_tool build_export <base_xbin> <dim> <num_vectors> <coarse> <fine> <batch_size> <codebook> <output_model>\n"
        << "  topoanns_rvq_tool verify <model> <query_xbin> <dim> <num_queries> <coarse> <fine>\n"
        << "  topoanns_rvq_tool bench_xbin <model> <query_bin> <dim> <num_queries> <coarse> <fine> <warmup_iters> <measure_iters>\n";
    std::exit(1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        Usage();
    }

    const std::string mode = argv[1];
    try {
        if (mode == "train_export") {
            if (argc != 10) {
                Usage();
            }
            TrainExport(argv[2],
                        std::stoi(argv[3]),
                        std::stoi(argv[4]),
                        std::stoi(argv[5]),
                        std::stoi(argv[6]),
                        std::stoi(argv[7]),
                        std::stoi(argv[8]),
                        argv[9]);
            return 0;
        }
        if (mode == "build_export") {
            if (argc != 10) {
                Usage();
            }
            BuildExport(argv[2],
                        std::stoi(argv[3]),
                        std::stoi(argv[4]),
                        std::stoi(argv[5]),
                        std::stoi(argv[6]),
                        std::stoi(argv[7]),
                        argv[8],
                        argv[9]);
            return 0;
        }
        if (mode == "verify") {
            if (argc != 8) {
                Usage();
            }
            Verify(argv[2],
                   argv[3],
                   std::stoi(argv[4]),
                   std::stoi(argv[5]),
                   std::stoi(argv[6]),
                   std::stoi(argv[7]));
            return 0;
        }
        if (mode == "bench_xbin") {
            if (argc != 10) {
                Usage();
            }
            BenchmarkXBin(argv[2],
                          argv[3],
                          std::stoi(argv[4]),
                          std::stoi(argv[5]),
                          std::stoi(argv[6]),
                          std::stoi(argv[7]),
                          std::stoi(argv[8]),
                          std::stoi(argv[9]));
            return 0;
        }
        Usage();
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_rvq] " << e.what() << std::endl;
        return 1;
    }
}
