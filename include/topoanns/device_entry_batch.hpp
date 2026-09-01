#pragma once

#include <cstddef>
#include <cstdint>

#include "topoanns/cuda_buffer.hpp"

namespace topoanns {

struct DeviceEntryBatch {
    DeviceEntryBatch() = default;

    DeviceEntryBatch(const DeviceEntryBatch&) = delete;
    DeviceEntryBatch& operator=(const DeviceEntryBatch&) = delete;

    DeviceEntryBatch(DeviceEntryBatch&&) noexcept = default;
    DeviceEntryBatch& operator=(DeviceEntryBatch&&) noexcept = default;

    std::size_t num_queries = 0;
    std::size_t entries_per_query = 0;
    CudaBuffer<std::uint32_t> offsets;
    CudaBuffer<std::uint32_t> ids;
};

}  // namespace topoanns
