#include "topoanns/pq_index.hpp"

#include <fcntl.h>
#include <cuda_runtime.h>

#include <cstring>
#include <fstream>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <unistd.h>

namespace topoanns {
namespace {

constexpr std::size_t kHostCodesRetainThresholdBytes = 1ULL << 30;
constexpr std::size_t kPinnedReadChunkBytes = 64ULL << 20;

template <typename T>
struct BinBlock {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> data;
};

template <typename T>
BinBlock<T> ReadBinBlock(const std::filesystem::path& path, std::size_t offset = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadBinBlock", "Failed to open " + path.string()));
    }

    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::int32_t rows_i32 = 0;
    std::int32_t cols_i32 = 0;
    in.read(reinterpret_cast<char*>(&rows_i32), sizeof(rows_i32));
    in.read(reinterpret_cast<char*>(&cols_i32), sizeof(cols_i32));
    if (!in.good() || rows_i32 < 0 || cols_i32 < 0) {
        throw std::runtime_error(BuildErrorMessage("ReadBinBlock",
                                                   "Invalid bin metadata in " + path.string()));
    }

    BinBlock<T> block;
    block.rows = static_cast<std::size_t>(rows_i32);
    block.cols = static_cast<std::size_t>(cols_i32);
    block.data.resize(block.rows * block.cols);
    if (!block.data.empty()) {
        in.read(reinterpret_cast<char*>(block.data.data()),
                static_cast<std::streamsize>(block.data.size() * sizeof(T)));
        if (!in.good()) {
            throw std::runtime_error(BuildErrorMessage("ReadBinBlock",
                                                       "Short read in " + path.string()));
        }
    }
    return block;
}

std::pair<std::size_t, std::size_t> ReadBinMetadata(const std::filesystem::path& path,
                                                    std::size_t offset) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadBinMetadata", "Failed to open " + path.string()));
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::int32_t rows_i32 = 0;
    std::int32_t cols_i32 = 0;
    in.read(reinterpret_cast<char*>(&rows_i32), sizeof(rows_i32));
    in.read(reinterpret_cast<char*>(&cols_i32), sizeof(cols_i32));
    if (!in.good() || rows_i32 < 0 || cols_i32 < 0) {
        throw std::runtime_error(BuildErrorMessage(
            "ReadBinMetadata", "Invalid metadata in " + path.string()));
    }
    return {static_cast<std::size_t>(rows_i32), static_cast<std::size_t>(cols_i32)};
}

std::vector<float> BuildColMajorTables(const std::vector<float>& row_major,
                                       std::size_t rows,
                                       std::size_t cols) {
    std::vector<float> col_major(row_major.size(), 0.0f);
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t col = 0; col < cols; ++col) {
            col_major[col * rows + row] = row_major[row * cols + col];
        }
    }
    return col_major;
}

void ReadExactly(int fd,
                 void* dst,
                 std::size_t bytes,
                 off_t offset,
                 const char* context) {
    char* out = static_cast<char*>(dst);
    std::size_t remaining = bytes;
    off_t current_offset = offset;
    while (remaining > 0) {
        const ssize_t read_result = pread(fd, out, remaining, current_offset);
        if (read_result < 0) {
            throw std::runtime_error(
                BuildErrorMessage(context, "Failed to read from PQ codes file."));
        }
        if (read_result == 0) {
            throw std::runtime_error(
                BuildErrorMessage(context, "Unexpected EOF in PQ codes file."));
        }
        out += read_result;
        remaining -= static_cast<std::size_t>(read_result);
        current_offset += read_result;
    }
}

}  // namespace

PqIndex PqIndex::LoadImpl(const std::filesystem::path& pivots_path,
                          const std::filesystem::path& compressed_path,
                          bool map_codes,
                          bool load_codes) {
    const auto total_begin = std::chrono::steady_clock::now();
    PqIndex index;

    const auto metadata_begin = std::chrono::steady_clock::now();
    const auto [num_centroids, pq_dim] =
        ReadBinMetadata(pivots_path, kPqMetadataSizeBytes);
    if (num_centroids != kNumPqCentroids) {
        throw std::runtime_error(BuildErrorMessage(
            "PqIndex::LoadFromSeparatePaths", "Unexpected PQ centroid count."));
    }
    index.host_.ndims = pq_dim;

    const auto [compressed_rows, compressed_cols] = ReadBinMetadata(compressed_path, 0);
    index.host_.num_points = compressed_rows;
    index.host_.num_chunks = compressed_cols;
    const std::size_t total_codes =
        index.host_.num_points * index.host_.num_chunks;
    const std::size_t total_code_bytes = total_codes * sizeof(std::uint8_t);

    const BinBlock<std::size_t> offsets_block = ReadBinBlock<std::size_t>(pivots_path);
    if (offsets_block.rows != 4 && offsets_block.rows != 5) {
        throw std::runtime_error(BuildErrorMessage(
            "PqIndex::LoadFromSeparatePaths", "Unexpected PQ offset count."));
    }
    const bool use_old_filetype = offsets_block.rows == 5;
    const std::size_t tables_offset = offsets_block.data[0];
    const std::size_t centroid_offset = offsets_block.data[1];
    const std::size_t chunk_offsets_offset =
        offsets_block.data[use_old_filetype ? 3 : 2];

    const BinBlock<float> tables = ReadBinBlock<float>(pivots_path, tables_offset);
    if (tables.rows != kNumPqCentroids || tables.cols != index.host_.ndims) {
        throw std::runtime_error(BuildErrorMessage(
            "PqIndex::LoadFromSeparatePaths", "Unexpected PQ table shape."));
    }
    index.host_.tables_row_major = tables.data;
    index.host_.tables_col_major =
        BuildColMajorTables(index.host_.tables_row_major, tables.rows, tables.cols);

    const BinBlock<float> centroid = ReadBinBlock<float>(pivots_path, centroid_offset);
    if (centroid.rows != index.host_.ndims || centroid.cols != 1) {
        throw std::runtime_error(BuildErrorMessage(
            "PqIndex::LoadFromSeparatePaths", "Unexpected PQ centroid vector shape."));
    }
    index.host_.centroid = centroid.data;

    const BinBlock<std::uint32_t> chunk_offsets =
        ReadBinBlock<std::uint32_t>(pivots_path, chunk_offsets_offset);
    if (chunk_offsets.cols != 1 ||
        chunk_offsets.rows != index.host_.num_chunks + 1) {
        throw std::runtime_error(BuildErrorMessage(
            "PqIndex::LoadFromSeparatePaths", "Unexpected PQ chunk offsets shape."));
    }
    index.host_.chunk_offsets = chunk_offsets.data;
    const auto metadata_end = std::chrono::steady_clock::now();

    const auto codes_begin = std::chrono::steady_clock::now();
    if (load_codes) {
    const int fd = open(compressed_path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(BuildErrorMessage("PqIndex::LoadFromSeparatePaths",
                                                   "Failed to open PQ codes file."));
    }

    std::size_t copied = 0;
    try {
        if (map_codes) {
            ThrowIfCudaError(cudaMallocHost(reinterpret_cast<void**>(&index.mapped_host_codes_),
                                            total_code_bytes),
                             "cudaMallocHost");
            index.host_.codes_on_host = true;
            while (copied < total_code_bytes) {
                const std::size_t chunk_bytes =
                    std::min(kPinnedReadChunkBytes, total_code_bytes - copied);
                ReadExactly(fd, index.mapped_host_codes_ + copied, chunk_bytes,
                            static_cast<off_t>(sizeof(std::int32_t) * 2 + copied),
                            "PqIndex::LoadFromSeparatePathsMapped");
                copied += chunk_bytes;
            }
            ThrowIfCudaError(
                cudaHostGetDevicePointer(reinterpret_cast<void**>(&index.mapped_device_codes_),
                                         index.mapped_host_codes_, 0),
                "cudaHostGetDevicePointer");
        } else {
            index.device_.codes = CudaBuffer<std::uint8_t>::Allocate(total_codes);
            index.host_.codes_on_host = total_code_bytes <= kHostCodesRetainThresholdBytes;
            if (index.host_.codes_on_host) {
                index.host_.codes.resize(total_codes);
            }

            void* staging_buffer = nullptr;
            ThrowIfCudaError(cudaMallocHost(&staging_buffer, kPinnedReadChunkBytes),
                             "cudaMallocHost");
            try {
                while (copied < total_code_bytes) {
                    const std::size_t chunk_bytes =
                        std::min(kPinnedReadChunkBytes, total_code_bytes - copied);
                    ReadExactly(fd, staging_buffer, chunk_bytes,
                                static_cast<off_t>(sizeof(std::int32_t) * 2 + copied),
                                "PqIndex::LoadFromSeparatePaths");
                    ThrowIfCudaError(
                        cudaMemcpy(index.device_.codes.get() + copied,
                                   staging_buffer,
                                   chunk_bytes,
                                   cudaMemcpyHostToDevice),
                        "cudaMemcpyHostToDevice");
                    if (index.host_.codes_on_host) {
                        std::memcpy(index.host_.codes.data() + copied, staging_buffer, chunk_bytes);
                    }
                    copied += chunk_bytes;
                }
            } catch (...) {
                cudaFreeHost(staging_buffer);
                throw;
            }
            cudaFreeHost(staging_buffer);
        }
    } catch (...) {
        close(fd);
        index.Release();
        throw;
    }
    close(fd);
    }
    const auto codes_end = std::chrono::steady_clock::now();

    const auto small_buffers_begin = std::chrono::steady_clock::now();
    index.device_.tables_col_major =
        CudaBuffer<float>::CopyFromHost(index.host_.tables_col_major);
    index.device_.chunk_offsets =
        CudaBuffer<std::uint32_t>::CopyFromHost(index.host_.chunk_offsets);
    index.device_.centroid = CudaBuffer<float>::CopyFromHost(index.host_.centroid);
    const auto small_buffers_end = std::chrono::steady_clock::now();
    const auto total_end = std::chrono::steady_clock::now();
    std::cout << "[topoanns_pq_load]"
              << " code_bytes=" << total_code_bytes
              << " retain_host_codes=" << (index.host_.codes_on_host ? 1 : 0)
              << " mapped_codes=" << (map_codes ? 1 : 0)
              << " metadata_tables_ms="
              << std::chrono::duration<double, std::milli>(metadata_end - metadata_begin).count()
              << " codes_to_gpu_ms="
              << std::chrono::duration<double, std::milli>(codes_end - codes_begin).count()
              << " small_buffers_ms="
              << std::chrono::duration<double, std::milli>(small_buffers_end - small_buffers_begin)
                     .count()
              << " total_ms="
              << std::chrono::duration<double, std::milli>(total_end - total_begin).count()
              << std::endl;
    return index;
}

PqIndex::~PqIndex() {
    Release();
}

PqIndex::PqIndex(PqIndex&& other) noexcept {
    *this = std::move(other);
}

PqIndex& PqIndex::operator=(PqIndex&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    Release();
    host_ = std::move(other.host_);
    device_ = std::move(other.device_);
    mapped_host_codes_ = other.mapped_host_codes_;
    mapped_device_codes_ = other.mapped_device_codes_;
    other.mapped_host_codes_ = nullptr;
    other.mapped_device_codes_ = nullptr;
    return *this;
}

PqIndex PqIndex::LoadFromSeparatePaths(const std::filesystem::path& pivots_path,
                                       const std::filesystem::path& compressed_path) {
    return LoadImpl(pivots_path, compressed_path, false, true);
}

PqIndex PqIndex::LoadFromSeparatePathsMapped(const std::filesystem::path& pivots_path,
                                             const std::filesystem::path& compressed_path) {
    return LoadImpl(pivots_path, compressed_path, true, true);
}

PqIndex PqIndex::LoadPivotsOnly(const std::filesystem::path& pivots_path,
                                const std::filesystem::path& compressed_path) {
    return LoadImpl(pivots_path, compressed_path, false, false);
}

const std::uint8_t* PqIndex::host_codes() const {
    if (mapped_host_codes_ != nullptr) {
        return mapped_host_codes_;
    }
    return host_.codes.empty() ? nullptr : host_.codes.data();
}

const std::uint8_t* PqIndex::device_codes() const {
    if (mapped_device_codes_ != nullptr) {
        return mapped_device_codes_;
    }
    return device_.codes.get();
}

void PqIndex::Release() noexcept {
    if (mapped_host_codes_ != nullptr) {
        cudaFreeHost(mapped_host_codes_);
    }
    mapped_host_codes_ = nullptr;
    mapped_device_codes_ = nullptr;
}

}  // namespace topoanns
