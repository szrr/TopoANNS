#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "topoanns/common.hpp"
#include "topoanns/vector_page_layout.hpp"

namespace topoanns {

struct VectorStoreHeader {
    char magic[8];
    std::uint32_t version;
    std::uint32_t scalar_kind;
    std::uint32_t dim;
    std::uint32_t reserved0;
    std::uint64_t num_vectors;
    std::uint64_t page_size_bytes;
    std::uint64_t vector_bytes;
    std::uint64_t vectors_per_page;
    std::uint8_t reserved[kDefaultPageSizeBytes - 56];
};

static_assert(sizeof(VectorStoreHeader) == kDefaultPageSizeBytes,
              "VectorStoreHeader must be exactly one 4KB page.");

class VectorStoreBuilder {
public:
    static constexpr std::uint32_t kVersion = 1;

    static void WriteFile(const std::filesystem::path& path,
                          const VectorPageLayout& layout,
                          ScalarKind scalar_kind,
                          std::uint32_t dim,
                          const std::vector<std::vector<std::uint8_t>>& vectors);
    static VectorStoreHeader ReadHeader(const std::filesystem::path& path);
    static std::vector<std::uint8_t> ReadVectorById(
        const std::filesystem::path& path,
        const VectorPageLayout& layout,
        std::uint64_t node_id);
};

}  // namespace topoanns
