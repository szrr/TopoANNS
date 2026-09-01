#include "topoanns/pinned_topology.hpp"

#include <cuda_runtime.h>

#include <fcntl.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <unistd.h>
#include <utility>

namespace topoanns {
namespace {

void CheckCuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        throw std::runtime_error(BuildErrorMessage(context, cudaGetErrorString(status)));
    }
}

__global__ void copy_node_neighbors_kernel(const std::uint32_t* topology,
                                           std::uint64_t base_index,
                                           std::uint32_t degree,
                                           std::uint32_t* out) {
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < degree) {
        out[idx] = topology[base_index + idx];
    }
}

}  // namespace

PinnedTopology::~PinnedTopology() {
    Release();
}

PinnedTopology::PinnedTopology(PinnedTopology&& other) noexcept {
    *this = std::move(other);
}

PinnedTopology& PinnedTopology::operator=(PinnedTopology&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    Release();
    host_data_ = other.host_data_;
    device_data_ = other.device_data_;
    num_nodes_ = other.num_nodes_;
    degree_ = other.degree_;
    element_count_ = other.element_count_;
    other.host_data_ = nullptr;
    other.device_data_ = nullptr;
    other.num_nodes_ = 0;
    other.degree_ = 0;
    other.element_count_ = 0;
    return *this;
}

PinnedTopology PinnedTopology::FromDataset(const TopologyDataset& dataset) {
    return FromRaw(dataset.num_nodes, dataset.degree, dataset.adjacency);
}

PinnedTopology PinnedTopology::FromFile(const std::filesystem::path& path) {
    const auto total_begin = std::chrono::steady_clock::now();
    const TopologyHeader header = TopologyLayout::ReadHeader(path);
    const std::size_t element_count =
        static_cast<std::size_t>(header.num_nodes) * header.degree;
    const std::size_t bytes = element_count * sizeof(std::uint32_t);

    PinnedTopology pinned;
    pinned.num_nodes_ = header.num_nodes;
    pinned.degree_ = header.degree;
    pinned.element_count_ = element_count;

    const auto alloc_begin = std::chrono::steady_clock::now();
    CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&pinned.host_data_), bytes),
              "cudaMallocHost");
    const auto alloc_end = std::chrono::steady_clock::now();

    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        pinned.Release();
        throw std::runtime_error(
            BuildErrorMessage("PinnedTopology::FromFile", "Failed to open topology file."));
    }

    char* dst = reinterpret_cast<char*>(pinned.host_data_);
    std::size_t remaining = bytes;
    std::size_t offset = sizeof(TopologyHeader);
    const auto read_begin = std::chrono::steady_clock::now();
    while (remaining > 0) {
        const ssize_t read_result = pread(fd, dst, remaining, static_cast<off_t>(offset));
        if (read_result < 0) {
            close(fd);
            pinned.Release();
            throw std::runtime_error(BuildErrorMessage("PinnedTopology::FromFile",
                                                       "Failed to read topology payload."));
        }
        if (read_result == 0) {
            close(fd);
            pinned.Release();
            throw std::runtime_error(BuildErrorMessage("PinnedTopology::FromFile",
                                                       "Unexpected EOF while reading topology."));
        }
        dst += read_result;
        remaining -= static_cast<std::size_t>(read_result);
        offset += static_cast<std::size_t>(read_result);
    }
    close(fd);
    const auto read_end = std::chrono::steady_clock::now();

    const auto uva_begin = std::chrono::steady_clock::now();
    CheckCuda(cudaHostGetDevicePointer(
                  reinterpret_cast<void**>(&pinned.device_data_),
                  pinned.host_data_, 0),
              "cudaHostGetDevicePointer");
    const auto uva_end = std::chrono::steady_clock::now();
    const auto total_end = std::chrono::steady_clock::now();
    std::cout << "[topoanns_topology_load]"
              << " bytes=" << bytes
              << " alloc_ms="
              << std::chrono::duration<double, std::milli>(alloc_end - alloc_begin).count()
              << " read_ms="
              << std::chrono::duration<double, std::milli>(read_end - read_begin).count()
              << " uva_map_ms="
              << std::chrono::duration<double, std::milli>(uva_end - uva_begin).count()
              << " total_ms="
              << std::chrono::duration<double, std::milli>(total_end - total_begin).count()
              << std::endl;
    return pinned;
}

PinnedTopology PinnedTopology::FromRaw(std::uint64_t num_nodes,
                                       std::uint32_t degree,
                                       const std::vector<std::uint32_t>& adjacency) {
    if (degree == 0) {
        throw std::runtime_error(BuildErrorMessage("FromRaw", "degree must be positive."));
    }
    const std::size_t expected_size = static_cast<std::size_t>(num_nodes) * degree;
    if (adjacency.size() != expected_size) {
        throw std::runtime_error(BuildErrorMessage(
            "FromRaw", "adjacency size must equal num_nodes * degree."));
    }

    PinnedTopology pinned;
    pinned.num_nodes_ = num_nodes;
    pinned.degree_ = degree;
    pinned.element_count_ = adjacency.size();

    const std::size_t bytes = adjacency.size() * sizeof(std::uint32_t);
    CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&pinned.host_data_), bytes),
              "cudaMallocHost");
    std::memcpy(pinned.host_data_, adjacency.data(), bytes);
    CheckCuda(cudaHostGetDevicePointer(
                  reinterpret_cast<void**>(&pinned.device_data_),
                  pinned.host_data_, 0),
              "cudaHostGetDevicePointer");
    return pinned;
}

std::vector<std::uint32_t> PinnedTopology::ReadHostNode(std::uint64_t node_id) const {
    if (node_id >= num_nodes_) {
        throw std::runtime_error(BuildErrorMessage("ReadHostNode", "node_id is out of range."));
    }
    const std::uint32_t* src = host_data_ + node_id * degree_;
    return std::vector<std::uint32_t>(src, src + degree_);
}

std::vector<std::uint32_t> PinnedTopology::CopyNodeFromDevice(std::uint64_t node_id) const {
    if (node_id >= num_nodes_) {
        throw std::runtime_error(
            BuildErrorMessage("CopyNodeFromDevice", "node_id is out of range."));
    }
    std::uint32_t* device_output = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(degree_) * sizeof(std::uint32_t);
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_output), bytes), "cudaMalloc");

    const std::uint64_t base_index = node_id * degree_;
    const int threads = 128;
    const int blocks = static_cast<int>((degree_ + threads - 1) / threads);
    copy_node_neighbors_kernel<<<blocks, threads>>>(device_data_, base_index, degree_,
                                                    device_output);
    CheckCuda(cudaGetLastError(), "copy_node_neighbors_kernel");
    CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    std::vector<std::uint32_t> host_output(degree_);
    CheckCuda(cudaMemcpy(host_output.data(), device_output, bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpyDeviceToHost");
    CheckCuda(cudaFree(device_output), "cudaFree");
    return host_output;
}

void PinnedTopology::Release() noexcept {
    if (host_data_ != nullptr) {
        cudaFreeHost(host_data_);
    }
    host_data_ = nullptr;
    device_data_ = nullptr;
    num_nodes_ = 0;
    degree_ = 0;
    element_count_ = 0;
}

}  // namespace topoanns
