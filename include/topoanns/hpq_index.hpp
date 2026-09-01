#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>

#include "topoanns/pq_index.hpp"

namespace topoanns {

// DGAI HPQ stores one hybrid code per subspace and a bit selecting the base or
// outlier codebook. The large arrays remain in mapped pinned host memory.
class HpqIndex {
public:
    HpqIndex() = default;
    ~HpqIndex();

    HpqIndex(HpqIndex&& other) noexcept;
    HpqIndex& operator=(HpqIndex&& other) noexcept;

    HpqIndex(const HpqIndex&) = delete;
    HpqIndex& operator=(const HpqIndex&) = delete;

    static HpqIndex Load(const std::filesystem::path& base_pivots_path,
                         const std::filesystem::path& outlier_pivots_path,
                         const std::filesystem::path& hybrid_codes_path,
                         const std::filesystem::path& selector_bits_path);

    const PqIndex& base_index() const { return base_index_; }
    const PqIndex& outlier_index() const { return outlier_index_; }
    std::size_t selector_stride_bytes() const { return selector_stride_bytes_; }
    const std::uint8_t* device_selector_bits() const { return mapped_device_selector_bits_; }

private:
    void Release() noexcept;

    PqIndex base_index_;
    PqIndex outlier_index_;
    std::size_t selector_stride_bytes_ = 0;
    std::uint8_t* mapped_host_selector_bits_ = nullptr;
    std::uint8_t* mapped_device_selector_bits_ = nullptr;
};

}  // namespace topoanns
