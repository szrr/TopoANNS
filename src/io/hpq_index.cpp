#include "topoanns/hpq_index.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace topoanns {
namespace {

constexpr std::size_t kPinnedReadChunkBytes = 64ULL << 20;

void ReadExactly(int fd, void* destination, std::size_t bytes, off_t offset) {
    auto* out = static_cast<std::uint8_t*>(destination);
    while (bytes != 0) {
        const ssize_t count = pread(fd, out, bytes, offset);
        if (count <= 0) {
            throw std::runtime_error(
                BuildErrorMessage("HpqIndex::Load", "Short selector-bit read."));
        }
        out += count;
        bytes -= static_cast<std::size_t>(count);
        offset += count;
    }
}

}  // namespace

HpqIndex::~HpqIndex() {
    Release();
}

HpqIndex::HpqIndex(HpqIndex&& other) noexcept {
    *this = std::move(other);
}

HpqIndex& HpqIndex::operator=(HpqIndex&& other) noexcept {
    if (this != &other) {
        Release();
        base_index_ = std::move(other.base_index_);
        outlier_index_ = std::move(other.outlier_index_);
        selector_stride_bytes_ = other.selector_stride_bytes_;
        mapped_host_selector_bits_ = other.mapped_host_selector_bits_;
        mapped_device_selector_bits_ = other.mapped_device_selector_bits_;
        other.selector_stride_bytes_ = 0;
        other.mapped_host_selector_bits_ = nullptr;
        other.mapped_device_selector_bits_ = nullptr;
    }
    return *this;
}

HpqIndex HpqIndex::Load(const std::filesystem::path& base_pivots_path,
                        const std::filesystem::path& outlier_pivots_path,
                        const std::filesystem::path& hybrid_codes_path,
                        const std::filesystem::path& selector_bits_path) {
    HpqIndex result;
    result.base_index_ =
        PqIndex::LoadPivotsOnly(base_pivots_path, hybrid_codes_path);
    result.outlier_index_ =
        PqIndex::LoadFromSeparatePathsMapped(outlier_pivots_path, hybrid_codes_path);

    std::ifstream metadata(selector_bits_path, std::ios::binary);
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    metadata.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    metadata.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!metadata.good() || rows < 0 || cols <= 0) {
        throw std::runtime_error(
            BuildErrorMessage("HpqIndex::Load", "Invalid selector-bit metadata."));
    }
    metadata.close();

    const auto& host = result.outlier_index_.host();
    const std::size_t expected_stride = (host.num_chunks + 7U) / 8U;
    if (static_cast<std::uint64_t>(rows) != host.num_points ||
        static_cast<std::size_t>(cols) != expected_stride) {
        throw std::runtime_error(BuildErrorMessage(
            "HpqIndex::Load", "Selector-bit shape does not match HPQ codes."));
    }
    result.selector_stride_bytes_ = expected_stride;
    const std::size_t total_bytes = static_cast<std::size_t>(rows) * expected_stride;
    const int fd = open(selector_bits_path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error(
            BuildErrorMessage("HpqIndex::Load", "Failed to open selector-bit file."));
    }
    try {
        ThrowIfCudaError(cudaMallocHost(
                             reinterpret_cast<void**>(&result.mapped_host_selector_bits_),
                             total_bytes),
                         "cudaMallocHost(HPQ selectors)");
        std::size_t copied = 0;
        while (copied < total_bytes) {
            const std::size_t chunk = std::min(kPinnedReadChunkBytes, total_bytes - copied);
            ReadExactly(fd, result.mapped_host_selector_bits_ + copied, chunk,
                        static_cast<off_t>(sizeof(std::int32_t) * 2 + copied));
            copied += chunk;
        }
        ThrowIfCudaError(cudaHostGetDevicePointer(
                             reinterpret_cast<void**>(&result.mapped_device_selector_bits_),
                             result.mapped_host_selector_bits_, 0),
                         "cudaHostGetDevicePointer(HPQ selectors)");
    } catch (...) {
        close(fd);
        result.Release();
        throw;
    }
    close(fd);
    std::cout << "[topoanns_hpq_load]"
              << " code_bytes=" << host.num_points * host.num_chunks
              << " selector_bytes=" << total_bytes
              << " chunks=" << host.num_chunks << std::endl;
    return result;
}

void HpqIndex::Release() noexcept {
    if (mapped_host_selector_bits_ != nullptr) {
        cudaFreeHost(mapped_host_selector_bits_);
    }
    mapped_host_selector_bits_ = nullptr;
    mapped_device_selector_bits_ = nullptr;
    selector_stride_bytes_ = 0;
}

}  // namespace topoanns
