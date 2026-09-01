#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "topoanns/common.hpp"

namespace topoanns {

struct BvecsMetadata {
    std::uint32_t dim = 0;
    std::uint32_t record_bytes = 0;
    std::uint64_t num_vectors = 0;
};

BvecsMetadata ReadBvecsMetadata(const std::filesystem::path& path);

std::vector<float> ReadBvecsRangeAsFloat32(const std::filesystem::path& path,
                                           std::uint64_t start_vector,
                                           std::uint64_t num_vectors);

}  // namespace topoanns
