#include "topoanns/rvq_entry_provider.hpp"

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "RVQ/RVQ.cuh"
#include "rvq_device_entries.cuh"
#include "topoanns/common.hpp"
#include "topoanns/cuda_buffer.hpp"

namespace topoanns {
namespace {

constexpr std::size_t kGpuClusterPostingLimit = 256;

struct RvqClusterAssignProfile {
    double query_upload_ms = 0.0;
    double search_kernel_ms = 0.0;
};

struct RvqClusterAssignDeviceResult {
    CudaBuffer<int> device_clusters;
    double query_upload_ms = 0.0;
    double search_kernel_ms = 0.0;
};

struct RvqModelHeader {
    std::int32_t dim = 0;
    std::int32_t coarse_centroids = 0;
    std::int32_t fine_centroids = 0;
};

RvqModelHeader ReadRvqModelHeader(const std::filesystem::path& model_path) {
    std::ifstream in(model_path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadRvqModelHeader", "Failed to open " + model_path.string()));
    }

    RvqModelHeader header;
    in.read(reinterpret_cast<char*>(&header.dim), sizeof(header.dim));
    in.read(reinterpret_cast<char*>(&header.coarse_centroids),
            sizeof(header.coarse_centroids));
    in.read(reinterpret_cast<char*>(&header.fine_centroids), sizeof(header.fine_centroids));
    if (!in.good() || header.dim <= 0 || header.coarse_centroids <= 0 ||
        header.fine_centroids <= 0) {
        throw std::runtime_error(BuildErrorMessage("ReadRvqModelHeader",
                                                   "Invalid RVQ model header in " +
                                                       model_path.string()));
    }
    return header;
}

RvqClusterAssignDeviceResult AssignClustersFloat32Device(
    const RVQ& rvq,
    const std::vector<float>& queries,
    std::size_t num_queries,
    std::size_t dim,
    RvqClusterAssignProfile* out_profile) {
    if (queries.size() != num_queries * dim) {
        throw std::runtime_error(BuildErrorMessage(
            "AssignClustersFloat32", "queries size must equal num_queries * dim."));
    }

    const auto upload_begin = std::chrono::steady_clock::now();
    CudaBuffer<float> device_queries = CudaBuffer<float>::CopyFromHost(queries);
    const auto upload_end = std::chrono::steady_clock::now();
    CudaBuffer<int> device_clusters = CudaBuffer<int>::Allocate(num_queries);

    cudaEvent_t kernel_begin = nullptr;
    cudaEvent_t kernel_end = nullptr;
    ThrowIfCudaError(cudaEventCreate(&kernel_begin), "cudaEventCreate");
    ThrowIfCudaError(cudaEventCreate(&kernel_end), "cudaEventCreate");
    ThrowIfCudaError(cudaEventRecord(kernel_begin), "cudaEventRecord");
    const_cast<RVQ&>(rvq).search(device_queries.get(),
                                 static_cast<int>(num_queries),
                                 device_clusters.get());
    ThrowIfCudaError(cudaGetLastError(), "RVQ::search");
    ThrowIfCudaError(cudaEventRecord(kernel_end), "cudaEventRecord");
    ThrowIfCudaError(cudaEventSynchronize(kernel_end), "cudaEventSynchronize");
    float kernel_ms = 0.0f;
    ThrowIfCudaError(cudaEventElapsedTime(&kernel_ms, kernel_begin, kernel_end),
                     "cudaEventElapsedTime");
    cudaEventDestroy(kernel_begin);
    cudaEventDestroy(kernel_end);
    RvqClusterAssignDeviceResult result;
    result.device_clusters = std::move(device_clusters);
    result.query_upload_ms =
        std::chrono::duration<double, std::milli>(upload_end - upload_begin).count();
    result.search_kernel_ms = static_cast<double>(kernel_ms);
    if (out_profile != nullptr) {
        out_profile->query_upload_ms += result.query_upload_ms;
        out_profile->search_kernel_ms += result.search_kernel_ms;
    }
    return result;
}

std::vector<int> AssignClustersFloat32(const RVQ& rvq,
                                       const std::vector<float>& queries,
                                       std::size_t num_queries,
                                       std::size_t dim,
                                       RvqClusterAssignProfile* out_profile) {
    RvqClusterAssignDeviceResult device_result = AssignClustersFloat32Device(
        rvq, queries, num_queries, dim, out_profile);
    return device_result.device_clusters.CopyToHost();
}

std::pair<std::vector<std::uint32_t>, std::vector<std::uint32_t>> FlattenClusterPostings(
    const RVQ& rvq) {
    const int num_coarse_centroids = rvq.get_numCoarseCentroid();
    const int num_fine_centroids = rvq.get_numFineCentroid();
    const auto index = rvq.get_index();
    const std::size_t cluster_count =
        static_cast<std::size_t>(num_coarse_centroids) *
        static_cast<std::size_t>(num_fine_centroids);
    std::vector<std::uint32_t> offsets(cluster_count + 1, 0U);
    std::vector<std::uint32_t> flat_ids;
    std::size_t total_points = 0;
    for (std::size_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
        const int coarse_id = static_cast<int>(cluster_id / num_fine_centroids);
        const int fine_id = static_cast<int>(cluster_id % num_fine_centroids);
        total_points += index[coarse_id][fine_id].size();
    }
    flat_ids.reserve(total_points);
    for (std::size_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
        const int coarse_id = static_cast<int>(cluster_id / num_fine_centroids);
        const int fine_id = static_cast<int>(cluster_id % num_fine_centroids);
        const auto& bucket = index[coarse_id][fine_id];
        const std::size_t limit = std::min<std::size_t>(bucket.size(), kGpuClusterPostingLimit);
        for (std::size_t i = 0; i < limit; ++i) {
            const idx_t id = bucket[i];
            flat_ids.push_back(static_cast<std::uint32_t>(id));
        }
        offsets[cluster_id + 1] = static_cast<std::uint32_t>(flat_ids.size());
    }
    return {std::move(offsets), std::move(flat_ids)};
}

}  // namespace

struct RvqModel::Impl {
    explicit Impl(const std::filesystem::path& model_path) {
        const RvqModelHeader header = ReadRvqModelHeader(model_path);
        dim = header.dim;
        coarse_centroids = header.coarse_centroids;
        fine_centroids = header.fine_centroids;
        rvq = std::make_unique<RVQ>(header.dim,
                                    header.coarse_centroids,
                                    header.fine_centroids);
        rvq->load(model_path.string());
        const std::size_t cluster_count =
            static_cast<std::size_t>(header.coarse_centroids) *
            static_cast<std::size_t>(header.fine_centroids);
        const auto index = rvq->get_index();
        std::vector<int> cluster_sizes_host(cluster_count, 0);
        for (std::size_t cluster_id = 0; cluster_id < cluster_count; ++cluster_id) {
            const int coarse_id = static_cast<int>(cluster_id / header.fine_centroids);
            const int fine_id = static_cast<int>(cluster_id % header.fine_centroids);
            cluster_sizes_host[cluster_id] = static_cast<int>(index[coarse_id][fine_id].size());
        }
        cluster_sizes = CudaBuffer<int>::CopyFromHost(cluster_sizes_host);
        auto [cluster_offsets_host, cluster_ids_host] = FlattenClusterPostings(*rvq);
        cluster_offsets = CudaBuffer<std::uint32_t>::CopyFromHost(cluster_offsets_host);
        cluster_ids = CudaBuffer<std::uint32_t>::CopyFromHost(cluster_ids_host);
        WarmUpSearch();
    }

    void WarmUpSearch() {
        std::vector<float> warm_query(static_cast<std::size_t>(dim), 0.0f);
        CudaBuffer<float> device_query = CudaBuffer<float>::CopyFromHost(warm_query);
        CudaBuffer<int> device_cluster = CudaBuffer<int>::Allocate(1);
        rvq->search(device_query.get(), 1, device_cluster.get());
        ThrowIfCudaError(cudaGetLastError(), "RVQ::search warmup");
        ThrowIfCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    std::unique_ptr<RVQ> rvq;
    int dim = 0;
    int coarse_centroids = 0;
    int fine_centroids = 0;
    CudaBuffer<int> cluster_sizes;
    CudaBuffer<std::uint32_t> cluster_offsets;
    CudaBuffer<std::uint32_t> cluster_ids;
};

RvqModel::RvqModel() = default;

RvqModel::~RvqModel() = default;

RvqModel::RvqModel(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

RvqModel::RvqModel(RvqModel&&) noexcept = default;

RvqModel& RvqModel::operator=(RvqModel&&) noexcept = default;

RvqModel RvqModel::Load(const std::filesystem::path& model_path) {
    return RvqModel(std::make_unique<Impl>(model_path));
}

std::uint32_t RvqModel::dim() const {
    if (!impl_) {
        throw std::runtime_error(
            BuildErrorMessage("RvqModel::dim", "model is not loaded."));
    }
    return static_cast<std::uint32_t>(impl_->dim);
}

std::uint32_t RvqModel::coarse_centroids() const {
    if (!impl_) {
        throw std::runtime_error(BuildErrorMessage("RvqModel::coarse_centroids",
                                                   "model is not loaded."));
    }
    return static_cast<std::uint32_t>(impl_->coarse_centroids);
}

std::uint32_t RvqModel::fine_centroids() const {
    if (!impl_) {
        throw std::runtime_error(BuildErrorMessage("RvqModel::fine_centroids",
                                                   "model is not loaded."));
    }
    return static_cast<std::uint32_t>(impl_->fine_centroids);
}

void RvqModel::WarmUp() const {
    if (!impl_) {
        throw std::runtime_error(BuildErrorMessage("RvqModel::WarmUp",
                                                   "model is not loaded."));
    }
    impl_->WarmUpSearch();
}

std::vector<std::vector<std::uint32_t>> RvqModel::ComputeFloat32EntryPoints(
    const std::vector<float>& queries,
    std::size_t num_queries,
    std::size_t entry_count,
    std::uint32_t fallback_entry,
    RvqEntryProfile* out_profile) const {
    if (!impl_) {
        throw std::runtime_error(BuildErrorMessage(
            "RvqModel::ComputeFloat32EntryPoints", "model is not loaded."));
    }
    if (entry_count == 0) {
        throw std::runtime_error(BuildErrorMessage(
            "RvqModel::ComputeFloat32EntryPoints", "entry_count must be positive."));
    }

    const std::size_t dim = static_cast<std::size_t>(impl_->dim);
    const auto total_begin = std::chrono::steady_clock::now();
    RvqClusterAssignProfile assign_profile;
    const std::vector<int> clusters =
        AssignClustersFloat32(*impl_->rvq, queries, num_queries, dim,
                              &assign_profile);
    const auto index = impl_->rvq->get_index();
    std::vector<std::vector<std::uint32_t>> entries_by_query(num_queries);
    for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
        const int cluster_id = clusters[query_id];
        if (cluster_id < 0) {
            entries_by_query[query_id].push_back(fallback_entry);
            continue;
        }
        const int coarse_id = cluster_id / impl_->fine_centroids;
        const int fine_id = cluster_id % impl_->fine_centroids;
        const auto& bucket = index[coarse_id][fine_id];
        const std::size_t limit = std::min(entry_count, bucket.size());
        entries_by_query[query_id].reserve(limit == 0 ? 1 : limit);
        for (std::size_t i = 0; i < limit; ++i) {
            entries_by_query[query_id].push_back(static_cast<std::uint32_t>(bucket[i]));
        }
        if (entries_by_query[query_id].empty()) {
            entries_by_query[query_id].push_back(fallback_entry);
        }
    }
    if (out_profile != nullptr) {
        const auto total_end = std::chrono::steady_clock::now();
        out_profile->total_ms +=
            std::chrono::duration<double, std::milli>(total_end - total_begin).count();
        out_profile->query_upload_ms += assign_profile.query_upload_ms;
        out_profile->search_kernel_ms += assign_profile.search_kernel_ms;
    }
    return entries_by_query;
}

DeviceEntryBatch RvqModel::ComputeFloat32DeviceEntryBatch(
    const std::vector<float>& queries,
    std::size_t num_queries,
    std::size_t entry_count,
    std::uint32_t fallback_entry,
    RvqEntryProfile* out_profile) const {
    if (!impl_) {
        throw std::runtime_error(BuildErrorMessage(
            "RvqModel::ComputeFloat32DeviceEntryBatch", "model is not loaded."));
    }
    if (entry_count == 0) {
        throw std::runtime_error(BuildErrorMessage(
            "RvqModel::ComputeFloat32DeviceEntryBatch", "entry_count must be positive."));
    }

    const std::size_t dim = static_cast<std::size_t>(impl_->dim);
    RvqClusterAssignProfile assign_profile;
    RvqClusterAssignDeviceResult cluster_result = AssignClustersFloat32Device(
        *impl_->rvq, queries, num_queries, dim, &assign_profile);
    double gather_ms = 0.0;
    DeviceEntryBatch batch = detail::BuildDeviceEntryBatchFromRvqClusters(
        cluster_result.device_clusters, num_queries, entry_count, fallback_entry,
        impl_->cluster_offsets, impl_->cluster_ids, &gather_ms);
    if (out_profile != nullptr) {
        out_profile->query_upload_ms += assign_profile.query_upload_ms;
        out_profile->search_kernel_ms += assign_profile.search_kernel_ms;
        out_profile->entry_gather_ms += gather_ms;
        out_profile->total_ms += assign_profile.query_upload_ms +
                                 assign_profile.search_kernel_ms + gather_ms;
    }
    return batch;
}

RvqEntryProvider::RvqEntryProvider(std::vector<std::vector<std::uint32_t>> entries_by_query)
    : entries_by_query_(std::move(entries_by_query)) {}

RvqEntryProvider RvqEntryProvider::FromFloatQueries(const RvqModel& model,
                                                    const std::vector<float>& queries,
                                                    std::size_t num_queries,
                                                    std::size_t entry_count,
                                                    std::uint32_t fallback_entry,
                                                    RvqEntryProfile* out_profile) {
    return RvqEntryProvider(model.ComputeFloat32EntryPoints(queries, num_queries,
                                                            entry_count, fallback_entry,
                                                            out_profile));
}

std::vector<std::uint32_t> RvqEntryProvider::GetEntryPoints(std::size_t query_id) const {
    if (query_id >= entries_by_query_.size()) {
        throw std::runtime_error(BuildErrorMessage(
            "RvqEntryProvider::GetEntryPoints", "query_id is out of range."));
    }
    return entries_by_query_[query_id];
}

}  // namespace topoanns
