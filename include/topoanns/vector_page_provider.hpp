#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "topoanns/cuda_buffer.hpp"

namespace topoanns {

struct DevicePageReadResult {
    CudaBuffer<std::uint8_t> page_bytes;
    double io_ms = 0.0;
};

class VectorPageProvider {
public:
    virtual ~VectorPageProvider() = default;

    virtual bool SupportsDeviceReads() const { return false; }
    virtual const void* DeviceReadHandle() const { return nullptr; }

    virtual std::vector<std::uint8_t> ReadPages(const std::filesystem::path& path,
                                                const std::vector<std::uint64_t>& page_ids,
                                                std::size_t header_bytes,
                                                std::size_t page_size_bytes) const = 0;

    virtual DevicePageReadResult ReadPagesToDevice(const std::filesystem::path& path,
                                                   const std::vector<std::uint64_t>& page_ids,
                                                   std::size_t header_bytes,
                                                   std::size_t page_size_bytes) const;

    virtual DevicePageReadResult ReadPagesToDevice(const std::filesystem::path& path,
                                                   const CudaBuffer<std::uint64_t>& page_ids,
                                                   std::size_t num_pages,
                                                   std::size_t header_bytes,
                                                   std::size_t page_size_bytes) const;
};

}  // namespace topoanns
