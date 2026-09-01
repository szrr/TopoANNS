#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "topoanns/common.hpp"

namespace topoanns {

template <typename T>
class CudaBuffer {
public:
    CudaBuffer() = default;
    ~CudaBuffer() { Release(); }

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&& other) noexcept { *this = std::move(other); }

    CudaBuffer& operator=(CudaBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        Release();
        ptr_ = other.ptr_;
        count_ = other.count_;
        other.ptr_ = nullptr;
        other.count_ = 0;
        return *this;
    }

    static CudaBuffer Allocate(std::size_t count) {
        CudaBuffer buffer;
        buffer.count_ = count;
        if (count == 0) {
            return buffer;
        }
        ThrowIfCudaError(cudaMalloc(reinterpret_cast<void**>(&buffer.ptr_),
                                    count * sizeof(T)),
                         "cudaMalloc");
        return buffer;
    }

    static CudaBuffer CopyFromHost(const std::vector<T>& host_data) {
        CudaBuffer buffer = Allocate(host_data.size());
        if (!host_data.empty()) {
            ThrowIfCudaError(
                cudaMemcpy(buffer.ptr_, host_data.data(), host_data.size() * sizeof(T),
                           cudaMemcpyHostToDevice),
                "cudaMemcpyHostToDevice");
        }
        return buffer;
    }

    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    std::size_t size() const { return count_; }
    bool empty() const { return count_ == 0; }

    std::vector<T> CopyToHost() const {
        std::vector<T> host_data(count_);
        if (!host_data.empty()) {
            ThrowIfCudaError(
                cudaMemcpy(host_data.data(), ptr_, host_data.size() * sizeof(T),
                           cudaMemcpyDeviceToHost),
                "cudaMemcpyDeviceToHost");
        }
        return host_data;
    }

private:
    void Release() noexcept {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
        ptr_ = nullptr;
        count_ = 0;
    }

    T* ptr_ = nullptr;
    std::size_t count_ = 0;
};

}  // namespace topoanns
