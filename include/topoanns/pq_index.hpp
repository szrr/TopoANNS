#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "topoanns/common.hpp"
#include "topoanns/cuda_buffer.hpp"

namespace topoanns {

constexpr std::size_t kPqMetadataSizeBytes = 4096;
constexpr std::size_t kNumPqBits = 8;
constexpr std::size_t kNumPqCentroids = 1ULL << kNumPqBits;

struct PqHostIndex {
    std::vector<std::uint8_t> codes;
    std::vector<float> tables_row_major;
    std::vector<float> tables_col_major;
    std::vector<std::uint32_t> chunk_offsets;
    std::vector<float> centroid;
    std::uint64_t num_points = 0;
    std::uint64_t ndims = 0;
    std::uint64_t num_chunks = 0;
    bool codes_on_host = false;
};

struct PqDeviceIndex {
    CudaBuffer<std::uint8_t> codes;
    CudaBuffer<float> tables_col_major;
    CudaBuffer<std::uint32_t> chunk_offsets;
    CudaBuffer<float> centroid;
};

class PqIndex {
public:
    PqIndex() = default;
    ~PqIndex();

    PqIndex(PqIndex&& other) noexcept;
    PqIndex& operator=(PqIndex&& other) noexcept;

    PqIndex(const PqIndex&) = delete;
    PqIndex& operator=(const PqIndex&) = delete;

    static PqIndex LoadFromSeparatePaths(const std::filesystem::path& pivots_path,
                                         const std::filesystem::path& compressed_path);
    static PqIndex LoadFromSeparatePathsMapped(const std::filesystem::path& pivots_path,
                                               const std::filesystem::path& compressed_path);
    static PqIndex LoadPivotsOnly(const std::filesystem::path& pivots_path,
                                  const std::filesystem::path& compressed_path);

    const PqHostIndex& host() const { return host_; }
    const PqDeviceIndex& device() const { return device_; }
    const std::uint8_t* host_codes() const;
    const std::uint8_t* device_codes() const;
    bool codes_are_mapped() const { return mapped_device_codes_ != nullptr; }

private:
    static PqIndex LoadImpl(const std::filesystem::path& pivots_path,
                            const std::filesystem::path& compressed_path,
                            bool map_codes,
                            bool load_codes);
    void Release() noexcept;

    PqHostIndex host_;
    PqDeviceIndex device_;
    std::uint8_t* mapped_host_codes_ = nullptr;
    std::uint8_t* mapped_device_codes_ = nullptr;
};

}  // namespace topoanns
