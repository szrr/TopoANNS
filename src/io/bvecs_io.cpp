#include "topoanns/bvecs_io.hpp"

#include <fstream>
#include <stdexcept>

namespace topoanns {

BvecsMetadata ReadBvecsMetadata(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadBvecsMetadata", "Failed to open " + path.string()));
    }
    const std::streamsize file_size = in.tellg();
    if (file_size < static_cast<std::streamsize>(sizeof(std::uint32_t))) {
        throw std::runtime_error(
            BuildErrorMessage("ReadBvecsMetadata", "File is too small to contain a header."));
    }
    in.seekg(0, std::ios::beg);
    std::uint32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in.good() || dim == 0) {
        throw std::runtime_error(
            BuildErrorMessage("ReadBvecsMetadata", "Failed to read a valid dimension header."));
    }
    const std::uint32_t record_bytes =
        static_cast<std::uint32_t>(sizeof(std::uint32_t) + dim);
    if (file_size % record_bytes != 0) {
        throw std::runtime_error(BuildErrorMessage(
            "ReadBvecsMetadata", "File size is not divisible by record size."));
    }
    return BvecsMetadata{
        dim,
        record_bytes,
        static_cast<std::uint64_t>(file_size / record_bytes),
    };
}

std::vector<float> ReadBvecsRangeAsFloat32(const std::filesystem::path& path,
                                           std::uint64_t start_vector,
                                           std::uint64_t num_vectors) {
    const BvecsMetadata metadata = ReadBvecsMetadata(path);
    if (start_vector > metadata.num_vectors ||
        start_vector + num_vectors > metadata.num_vectors) {
        throw std::runtime_error(BuildErrorMessage(
            "ReadBvecsRangeAsFloat32", "Requested range is out of bounds."));
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadBvecsRangeAsFloat32", "Failed to open " + path.string()));
    }
    in.seekg(static_cast<std::streamoff>(start_vector * metadata.record_bytes), std::ios::beg);

    std::vector<float> out(static_cast<std::size_t>(num_vectors) * metadata.dim, 0.0f);
    std::vector<std::uint8_t> record(metadata.record_bytes, 0);
    for (std::uint64_t i = 0; i < num_vectors; ++i) {
        in.read(reinterpret_cast<char*>(record.data()),
                static_cast<std::streamsize>(record.size()));
        if (!in.good()) {
            throw std::runtime_error(BuildErrorMessage(
                "ReadBvecsRangeAsFloat32", "Short read while loading vectors."));
        }
        const std::uint32_t record_dim =
            *reinterpret_cast<const std::uint32_t*>(record.data());
        if (record_dim != metadata.dim) {
            throw std::runtime_error(BuildErrorMessage(
                "ReadBvecsRangeAsFloat32", "Encountered inconsistent per-record dimension."));
        }
        const std::uint8_t* payload = record.data() + sizeof(std::uint32_t);
        float* dst = out.data() + static_cast<std::size_t>(i) * metadata.dim;
        for (std::uint32_t dim_idx = 0; dim_idx < metadata.dim; ++dim_idx) {
            dst[dim_idx] = static_cast<float>(payload[dim_idx]);
        }
    }
    return out;
}

}  // namespace topoanns
