#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

#include "topoanns/cuda_buffer.hpp"
#include "topoanns/vector_page_provider.hpp"
#include "topoanns/vector_page_layout.hpp"

namespace topoanns {

struct BamVectorProviderOptions {
    std::filesystem::path controller_path = "/dev/libnvm0";
    std::size_t device_offset_bytes = 0;
    std::size_t page_cache_size_bytes = 16ULL << 20;
    std::uint32_t cuda_device = 0;
    std::uint32_t nvm_namespace = 1;
    std::size_t queue_depth = 256;
    std::size_t num_queues = 8;
    std::size_t max_ranges = 16;
    bool create_primary_range = true;
    std::optional<std::size_t> payload_bytes_override;
    std::size_t range_id_base = 0;
};

struct BamIoProfileSnapshot {
    std::uint64_t gpu_cache_hits = 0;
    std::uint64_t host_cache_hits = 0;
    std::uint64_t physical_reads = 0;
    std::uint64_t profiled_submissions = 0;
    std::uint64_t profiled_completions = 0;
    std::uint64_t current_outstanding = 0;
    std::vector<std::uint64_t> submit_depth_histogram;
};

struct BamTraceBenchmarkResult {
    double elapsed_ms = 0.0;
    std::uint64_t checksum = 0;
    std::vector<std::uint32_t> block_sm_ids;
};

struct BamTraceBenchmarkWorkspace {
    CudaBuffer<unsigned long long> device_checksum;
    CudaBuffer<std::uint32_t> device_block_sm_ids;
};

using BamTraceLaunchHostCallback = void (*)(void*);

enum class BamTraceRequestMapping {
    kWarpPerPage,
    kThreadPerPage,
};

class BamVectorPageProvider final : public VectorPageProvider {
public:
    class Impl;

    BamVectorPageProvider(const std::filesystem::path& vector_store_path,
                          std::size_t header_bytes,
                          std::size_t page_size_bytes,
                          const BamVectorProviderOptions& options = {});
    ~BamVectorPageProvider() override;

    BamVectorPageProvider(BamVectorPageProvider&&) noexcept;
    BamVectorPageProvider& operator=(BamVectorPageProvider&&) noexcept;

    BamVectorPageProvider(const BamVectorPageProvider&) = delete;
    BamVectorPageProvider& operator=(const BamVectorPageProvider&) = delete;

    bool SupportsDeviceReads() const override { return true; }
    const void* DeviceReadHandle() const override;
    void FlushDevicePageCache();
    void ResetDevicePageCacheRanges();
    void ResetIoDepthProfile(bool enabled);
    BamIoProfileSnapshot ReadIoProfile() const;
    std::shared_ptr<VectorPageProvider> CreateFileRangeProvider(
        const std::filesystem::path& path,
        std::size_t header_bytes,
        std::size_t page_size_bytes,
        const BamVectorProviderOptions& options = {}) const;

    std::vector<std::uint8_t> ReadPages(const std::filesystem::path& path,
                                        const std::vector<std::uint64_t>& page_ids,
                                        std::size_t header_bytes,
                                        std::size_t page_size_bytes) const override;

    DevicePageReadResult ReadPagesToDevice(const std::filesystem::path& path,
                                           const std::vector<std::uint64_t>& page_ids,
                                           std::size_t header_bytes,
                                           std::size_t page_size_bytes) const override;

    DevicePageReadResult ReadPagesToDevice(const std::filesystem::path& path,
                                           const CudaBuffer<std::uint64_t>& page_ids,
                                           std::size_t num_pages,
                                           std::size_t header_bytes,
                                           std::size_t page_size_bytes) const override;

private:
    std::shared_ptr<Impl> impl_;
};

void WriteVectorStorePayloadToBam(const std::filesystem::path& vector_store_path,
                                  std::size_t header_bytes,
                                  std::size_t page_size_bytes,
                                  const BamVectorProviderOptions& options = {});

void WriteFilePayloadToBam(const std::filesystem::path& path,
                           std::size_t header_bytes,
                           std::size_t page_size_bytes,
                           const BamVectorProviderOptions& options = {});

std::vector<std::uint8_t> ReadBamCombinedNodeRecords(
    const VectorPageProvider& provider,
    const std::vector<std::uint32_t>& node_ids,
    std::size_t node_bytes,
    std::size_t nodes_per_page);

double RunBamFusedExactDistanceFloatQueries(const VectorPageProvider& provider,
                                            ScalarKind vector_scalar_kind,
                                            const CudaBuffer<float>& device_queries,
                                            const CudaBuffer<std::uint64_t>& page_ids,
                                            const CudaBuffer<std::uint32_t>& slot_ids,
                                            const CudaBuffer<std::uint32_t>& node_ids,
                                            const CudaBuffer<std::uint32_t>& candidate_query_ids,
                                            std::size_t num_candidates,
                                            const VectorPageLayout& layout,
                                            std::size_t dim,
                                            CudaBuffer<float>* out_distances);

BamTraceBenchmarkResult RunBam4kTraceBenchmark(
    const VectorPageProvider& provider,
    const CudaBuffer<std::uint64_t>& device_page_ids,
    std::size_t start_index,
    std::size_t num_requests,
    std::size_t num_blocks,
    std::size_t warps_per_block,
    std::size_t page_size_bytes = 4096,
    BamTraceRequestMapping request_mapping =
        BamTraceRequestMapping::kWarpPerPage);

BamTraceBenchmarkWorkspace PrepareBamTraceBenchmarkWorkspace(
    std::size_t num_blocks);

BamTraceBenchmarkResult RunPreparedBam4kTraceBenchmark(
    const VectorPageProvider& provider,
    const CudaBuffer<std::uint64_t>& device_page_ids,
    std::size_t start_index,
    std::size_t num_requests,
    std::size_t num_blocks,
    std::size_t warps_per_block,
    BamTraceBenchmarkWorkspace* workspace,
    cudaStream_t stream,
    std::size_t page_size_bytes = 4096,
    BamTraceRequestMapping request_mapping =
        BamTraceRequestMapping::kWarpPerPage,
    BamTraceLaunchHostCallback launch_callback = nullptr,
    void* launch_callback_context = nullptr);

}  // namespace topoanns
