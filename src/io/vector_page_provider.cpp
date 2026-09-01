#include "topoanns/vector_page_provider.hpp"

#include <stdexcept>

#include "topoanns/common.hpp"

namespace topoanns {

DevicePageReadResult VectorPageProvider::ReadPagesToDevice(
    const std::filesystem::path&,
    const std::vector<std::uint64_t>&,
    std::size_t,
    std::size_t) const {
    throw std::runtime_error(BuildErrorMessage("VectorPageProvider::ReadPagesToDevice",
                                               "Device-direct reads are not supported."));
}

DevicePageReadResult VectorPageProvider::ReadPagesToDevice(
    const std::filesystem::path&,
    const CudaBuffer<std::uint64_t>&,
    std::size_t,
    std::size_t,
    std::size_t) const {
    throw std::runtime_error(BuildErrorMessage("VectorPageProvider::ReadPagesToDevice",
                                               "Device-direct reads are not supported."));
}

}  // namespace topoanns
