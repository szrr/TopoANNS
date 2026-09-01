#pragma once

#include <cstddef>
#include <cstdint>

#include "topoanns/common.hpp"

namespace topoanns {

struct VectorPageAddress {
    std::uint64_t page_id = 0;
    std::uint32_t slot_id = 0;
    std::uint64_t byte_offset = 0;
};

class VectorPageLayout {
public:
    static VectorPageLayout Create(std::uint32_t dim,
                                   ScalarKind scalar_kind,
                                   std::size_t page_size_bytes = kDefaultPageSizeBytes);
    static VectorPageLayout CreateFromVectorBytes(
        std::size_t vector_bytes,
        std::size_t page_size_bytes = kDefaultPageSizeBytes);

    std::size_t page_size_bytes() const { return page_size_bytes_; }
    std::size_t header_bytes() const { return header_bytes_; }
    std::size_t vector_bytes() const { return vector_bytes_; }
    std::size_t vectors_per_page() const { return vectors_per_page_; }
    std::size_t payload_bytes_per_page() const { return payload_bytes_per_page_; }

    std::uint64_t PageCountForVectors(std::uint64_t num_vectors) const;
    VectorPageAddress Resolve(std::uint64_t node_id) const;

private:
    std::size_t page_size_bytes_ = kDefaultPageSizeBytes;
    std::size_t header_bytes_ = kDefaultPageSizeBytes;
    std::size_t vector_bytes_ = 0;
    std::size_t vectors_per_page_ = 0;
    std::size_t payload_bytes_per_page_ = 0;
};

}  // namespace topoanns
