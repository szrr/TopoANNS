#include "topoanns/vector_page_layout.hpp"

#include <stdexcept>

namespace topoanns {

VectorPageLayout VectorPageLayout::Create(std::uint32_t dim,
                                          ScalarKind scalar_kind,
                                          std::size_t page_size_bytes) {
    const std::size_t vector_bytes =
        static_cast<std::size_t>(dim) * ScalarKindBytes(scalar_kind);
    return CreateFromVectorBytes(vector_bytes, page_size_bytes);
}

VectorPageLayout VectorPageLayout::CreateFromVectorBytes(
    std::size_t vector_bytes,
    std::size_t page_size_bytes) {
    if (page_size_bytes == 0) {
        throw std::runtime_error(
            BuildErrorMessage("CreateFromVectorBytes", "page_size_bytes must be positive."));
    }
    if (vector_bytes == 0) {
        throw std::runtime_error(
            BuildErrorMessage("CreateFromVectorBytes", "vector_bytes must be positive."));
    }

    const std::size_t vectors_per_page = page_size_bytes / vector_bytes;
    if (vectors_per_page == 0) {
        throw std::runtime_error(BuildErrorMessage(
            "CreateFromVectorBytes",
            "A full vector does not fit inside a single page."));
    }

    VectorPageLayout layout;
    layout.page_size_bytes_ = page_size_bytes;
    layout.header_bytes_ = page_size_bytes;
    layout.vector_bytes_ = vector_bytes;
    layout.vectors_per_page_ = vectors_per_page;
    layout.payload_bytes_per_page_ = vectors_per_page * vector_bytes;
    return layout;
}

std::uint64_t VectorPageLayout::PageCountForVectors(std::uint64_t num_vectors) const {
    if (num_vectors == 0) {
        return 0;
    }
    return (num_vectors + vectors_per_page_ - 1) / vectors_per_page_;
}

VectorPageAddress VectorPageLayout::Resolve(std::uint64_t node_id) const {
    VectorPageAddress address;
    address.page_id = node_id / vectors_per_page_;
    address.slot_id = static_cast<std::uint32_t>(node_id % vectors_per_page_);
    address.byte_offset = header_bytes_ + address.page_id * page_size_bytes_ +
                          static_cast<std::uint64_t>(address.slot_id) * vector_bytes_;
    return address;
}

}  // namespace topoanns
