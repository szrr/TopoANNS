#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "topoanns/common.hpp"
#include "topoanns/cuda_buffer.hpp"
#include "topoanns/vector_page_provider.hpp"

class FileBackedTestVectorPageProvider final : public topoanns::VectorPageProvider {
public:
    explicit FileBackedTestVectorPageProvider(const std::filesystem::path& path) {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            throw std::runtime_error("Failed to open vector store for test provider.");
        }
        in.seekg(0, std::ios::end);
        const std::streamsize size = in.tellg();
        in.seekg(0, std::ios::beg);
        bytes_.resize(static_cast<std::size_t>(size), 0);
        in.read(reinterpret_cast<char*>(bytes_.data()), size);
        if (in.gcount() != size) {
            throw std::runtime_error("Failed to read vector store for test provider.");
        }
    }

    bool SupportsDeviceReads() const override { return true; }

    std::vector<std::uint8_t> ReadPages(const std::filesystem::path&,
                                        const std::vector<std::uint64_t>& page_ids,
                                        std::size_t header_bytes,
                                        std::size_t page_size_bytes) const override {
        std::vector<std::uint8_t> pages(page_ids.size() * page_size_bytes, 0);
        for (std::size_t i = 0; i < page_ids.size(); ++i) {
            const std::size_t byte_offset =
                header_bytes + static_cast<std::size_t>(page_ids[i]) * page_size_bytes;
            if (byte_offset + page_size_bytes > bytes_.size()) {
                throw std::runtime_error(topoanns::BuildErrorMessage(
                    "FileBackedTestVectorPageProvider::ReadPages",
                    "Requested page exceeds test vector store size."));
            }
            std::copy_n(bytes_.data() + byte_offset, page_size_bytes,
                        pages.data() + i * page_size_bytes);
        }
        return pages;
    }

    topoanns::DevicePageReadResult ReadPagesToDevice(
        const std::filesystem::path& path,
        const std::vector<std::uint64_t>& page_ids,
        std::size_t header_bytes,
        std::size_t page_size_bytes) const override {
        topoanns::DevicePageReadResult result;
        result.page_bytes =
            topoanns::CudaBuffer<std::uint8_t>::CopyFromHost(ReadPages(path,
                                                                       page_ids,
                                                                       header_bytes,
                                                                       page_size_bytes));
        return result;
    }

    topoanns::DevicePageReadResult ReadPagesToDevice(
        const std::filesystem::path&,
        const topoanns::CudaBuffer<std::uint64_t>& page_ids,
        std::size_t num_pages,
        std::size_t header_bytes,
        std::size_t page_size_bytes) const override {
        std::vector<std::uint64_t> host_page_ids = page_ids.CopyToHost();
        host_page_ids.resize(num_pages);
        topoanns::DevicePageReadResult result;
        result.page_bytes =
            topoanns::CudaBuffer<std::uint8_t>::CopyFromHost(ReadPages(std::filesystem::path{},
                                                                       host_page_ids,
                                                                       header_bytes,
                                                                       page_size_bytes));
        return result;
    }

private:
    std::vector<std::uint8_t> bytes_;
};
