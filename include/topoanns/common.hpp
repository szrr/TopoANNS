#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace topoanns {

constexpr std::size_t kDefaultPageSizeBytes = 4096;
constexpr std::uint32_t kFixedTopologyDegree = 64;
constexpr std::uint32_t kInvalidNodeId = std::numeric_limits<std::uint32_t>::max();

enum class ScalarKind {
    kInt8,
    kUint8,
    kFloat32,
};

inline std::size_t ScalarKindBytes(ScalarKind kind) {
    switch (kind) {
        case ScalarKind::kInt8:
            return 1;
        case ScalarKind::kUint8:
            return 1;
        case ScalarKind::kFloat32:
            return 4;
    }
    throw std::runtime_error("Unsupported ScalarKind.");
}

inline const char* ScalarKindName(ScalarKind kind) {
    switch (kind) {
        case ScalarKind::kInt8:
            return "int8";
        case ScalarKind::kUint8:
            return "uint8";
        case ScalarKind::kFloat32:
            return "float32";
    }
    return "unknown";
}

inline std::string BuildErrorMessage(const std::string& context,
                                     const std::string& detail) {
    return context + ": " + detail;
}

inline void ThrowIfCudaError(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(BuildErrorMessage(context, cudaGetErrorString(status)));
    }
}

}  // namespace topoanns
