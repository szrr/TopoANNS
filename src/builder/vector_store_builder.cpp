#include "topoanns/vector_store_builder.hpp"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace topoanns {
namespace {

constexpr char kVectorStoreMagic[8] = {'T', 'P', 'V', 'E', 'C', 'T', 'O', 'R'};

VectorStoreHeader BuildHeader(const VectorPageLayout& layout,
                              ScalarKind scalar_kind,
                              std::uint32_t dim,
                              std::uint64_t num_vectors) {
    VectorStoreHeader header{};
    std::memcpy(header.magic, kVectorStoreMagic, sizeof(header.magic));
    header.version = VectorStoreBuilder::kVersion;
    header.scalar_kind = static_cast<std::uint32_t>(scalar_kind);
    header.dim = dim;
    header.num_vectors = num_vectors;
    header.page_size_bytes = layout.page_size_bytes();
    header.vector_bytes = layout.vector_bytes();
    header.vectors_per_page = layout.vectors_per_page();
    return header;
}

void ValidateVectors(const VectorPageLayout& layout,
                     const std::vector<std::vector<std::uint8_t>>& vectors) {
    for (std::size_t i = 0; i < vectors.size(); ++i) {
        if (vectors[i].size() != layout.vector_bytes()) {
            throw std::runtime_error(BuildErrorMessage(
                "ValidateVectors", "Vector byte width mismatch at index " + std::to_string(i)));
        }
    }
}

}  // namespace

void VectorStoreBuilder::WriteFile(
    const std::filesystem::path& path,
    const VectorPageLayout& layout,
    ScalarKind scalar_kind,
    std::uint32_t dim,
    const std::vector<std::vector<std::uint8_t>>& vectors) {
    ValidateVectors(layout, vectors);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("WriteFile", "Failed to open vector store file."));
    }

    const VectorStoreHeader header =
        BuildHeader(layout, scalar_kind, dim, vectors.size());
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const std::uint64_t page_count = layout.PageCountForVectors(vectors.size());
    std::vector<std::uint8_t> page(layout.page_size_bytes(), 0);

    for (std::uint64_t page_id = 0; page_id < page_count; ++page_id) {
        std::fill(page.begin(), page.end(), 0);
        for (std::size_t slot = 0; slot < layout.vectors_per_page(); ++slot) {
            const std::uint64_t vector_id = page_id * layout.vectors_per_page() + slot;
            if (vector_id >= vectors.size()) {
                break;
            }
            const std::size_t slot_offset = slot * layout.vector_bytes();
            std::memcpy(page.data() + slot_offset, vectors[vector_id].data(),
                        layout.vector_bytes());
        }
        out.write(reinterpret_cast<const char*>(page.data()),
                  static_cast<std::streamsize>(page.size()));
    }

    if (!out.good()) {
        throw std::runtime_error(
            BuildErrorMessage("WriteFile", "Failed while writing vector store."));
    }
}

VectorStoreHeader VectorStoreBuilder::ReadHeader(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadHeader", "Failed to open vector store file."));
    }

    VectorStoreHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadHeader", "Failed to read vector store header."));
    }
    if (std::memcmp(header.magic, kVectorStoreMagic, sizeof(header.magic)) != 0) {
        throw std::runtime_error(
            BuildErrorMessage("ReadHeader", "Unexpected vector store magic."));
    }
    if (header.version != kVersion) {
        throw std::runtime_error(
            BuildErrorMessage("ReadHeader", "Unexpected vector store version."));
    }
    return header;
}

std::vector<std::uint8_t> VectorStoreBuilder::ReadVectorById(
    const std::filesystem::path& path,
    const VectorPageLayout& layout,
    std::uint64_t node_id) {
    const VectorStoreHeader header = ReadHeader(path);
    if (node_id >= header.num_vectors) {
        throw std::runtime_error(BuildErrorMessage(
            "ReadVectorById", "node_id is out of range for the vector store."));
    }

    const VectorPageAddress address = layout.Resolve(node_id);
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadVectorById", "Failed to open vector store file."));
    }
    in.seekg(static_cast<std::streamoff>(address.byte_offset), std::ios::beg);

    std::vector<std::uint8_t> vector(layout.vector_bytes(), 0);
    in.read(reinterpret_cast<char*>(vector.data()),
            static_cast<std::streamsize>(vector.size()));
    if (!in.good()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadVectorById", "Failed to read vector payload."));
    }
    return vector;
}

}  // namespace topoanns
