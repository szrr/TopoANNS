#include "topoanns/search_resources.hpp"

#include <cuda_runtime.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace topoanns {
namespace {

template <typename T>
struct BinBlock {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> data;
};

template <typename T>
BinBlock<T> ReadBinBlock(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error(
            BuildErrorMessage("ReadBinBlock", "Failed to open " + path.string()));
    }

    std::int32_t rows_i32 = 0;
    std::int32_t cols_i32 = 0;
    in.read(reinterpret_cast<char*>(&rows_i32), sizeof(rows_i32));
    in.read(reinterpret_cast<char*>(&cols_i32), sizeof(cols_i32));
    if (!in.good() || rows_i32 < 0 || cols_i32 < 0) {
        throw std::runtime_error(BuildErrorMessage("ReadBinBlock",
                                                   "Invalid metadata in " + path.string()));
    }

    BinBlock<T> block;
    block.rows = static_cast<std::size_t>(rows_i32);
    block.cols = static_cast<std::size_t>(cols_i32);
    block.data.resize(block.rows * block.cols);
    if (!block.data.empty()) {
        in.read(reinterpret_cast<char*>(block.data.data()),
                static_cast<std::streamsize>(block.data.size() * sizeof(T)));
        if (!in.good()) {
            throw std::runtime_error(
                BuildErrorMessage("ReadBinBlock", "Short read in " + path.string()));
        }
    }
    return block;
}

std::vector<float> BuildPq2CrossTerms(const PqIndex& base_pq, const PqIndex& residual_pq) {
    const auto& base = base_pq.host();
    const auto& residual = residual_pq.host();
    if (base.ndims != residual.ndims) {
        throw std::runtime_error(BuildErrorMessage("BuildPq2CrossTerms",
                                                   "Base PQ and residual PQ dimensions differ."));
    }
    if (base.num_chunks != residual.num_chunks) {
        throw std::runtime_error(BuildErrorMessage("BuildPq2CrossTerms",
                                                   "Base PQ and residual PQ chunk counts differ."));
    }
    if (base.chunk_offsets != residual.chunk_offsets) {
        throw std::runtime_error(BuildErrorMessage(
            "BuildPq2CrossTerms", "Base PQ and residual PQ chunk offsets differ."));
    }

    std::vector<float> cross_terms(base.num_chunks * kNumPqCentroids * kNumPqCentroids, 0.0f);
    for (std::size_t chunk = 0; chunk < base.num_chunks; ++chunk) {
        const std::size_t dim_begin = base.chunk_offsets[chunk];
        const std::size_t dim_end = base.chunk_offsets[chunk + 1];
        float* chunk_cross_terms =
            cross_terms.data() + chunk * kNumPqCentroids * kNumPqCentroids;
        for (std::size_t base_center = 0; base_center < kNumPqCentroids; ++base_center) {
            for (std::size_t residual_center = 0; residual_center < kNumPqCentroids;
                 ++residual_center) {
                float dot = 0.0f;
                for (std::size_t dim = dim_begin; dim < dim_end; ++dim) {
                    const float base_value =
                        base.tables_row_major[base_center * base.ndims + dim] + base.centroid[dim];
                    const float residual_value = residual.tables_row_major
                                                     [residual_center * residual.ndims + dim] +
                                                 residual.centroid[dim];
                    dot += base_value * residual_value;
                }
                chunk_cross_terms[base_center * kNumPqCentroids + residual_center] = 2.0f * dot;
            }
        }
    }
    return cross_terms;
}

}  // namespace

SearchResources SearchResources::FromDataset(const TopologyDataset& dataset) {
    SearchResources resources;
    resources.topology_ = PinnedTopology::FromDataset(dataset);
    return resources;
}

SearchResources SearchResources::FromTopologyFile(const std::filesystem::path& path) {
    SearchResources resources;
    resources.topology_ = PinnedTopology::FromFile(path);
    return resources;
}

std::uint64_t SearchResources::num_nodes() const {
    return topology_.num_nodes();
}

std::uint32_t SearchResources::degree() const {
    return topology_.degree();
}

std::vector<std::uint32_t> SearchResources::ReadHostNeighbors(std::uint64_t node_id) const {
    return topology_.ReadHostNode(node_id);
}

void SearchResources::LoadPqIndex(const std::filesystem::path& pivots_path,
                                  const std::filesystem::path& compressed_path) {
    pq_index_ = std::make_shared<PqIndex>(
        PqIndex::LoadFromSeparatePaths(pivots_path, compressed_path));
}

const PqIndex& SearchResources::pq_index() const {
    if (!pq_index_) {
        throw std::runtime_error(
            BuildErrorMessage("SearchResources::pq_index", "PQ index is not loaded."));
    }
    return *pq_index_;
}

void SearchResources::LoadPq2Index(const std::filesystem::path& pivots_path,
                                   const std::filesystem::path& compressed_path,
                                   const std::filesystem::path& error_bounds_path) {
    pq2_index_ = std::make_shared<PqIndex>(
        PqIndex::LoadFromSeparatePathsMapped(pivots_path, compressed_path));
    pq2_aux_.reset();

    if (error_bounds_path.empty()) {
        return;
    }
    if (!pq_index_) {
        throw std::runtime_error(BuildErrorMessage(
            "SearchResources::LoadPq2Index",
            "Base PQ index must be loaded before residual PQ refine resources."));
    }

    auto aux = std::make_shared<Pq2AuxResource>();
    aux->residual_refine = true;
    aux->cross_terms = CudaBuffer<float>::CopyFromHost(BuildPq2CrossTerms(*pq_index_, *pq2_index_));

    const BinBlock<float> error_bounds = ReadBinBlock<float>(error_bounds_path);
    if (error_bounds.cols != 1 ||
        error_bounds.rows != static_cast<std::size_t>(pq2_index_->host().num_points)) {
        throw std::runtime_error(BuildErrorMessage(
            "SearchResources::LoadPq2Index",
            "Residual error-bound file shape does not match PQ2 point count."));
    }

    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    ThrowIfCudaError(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
    const std::size_t cross_term_bytes = aux->cross_terms.size() * sizeof(float);
    const std::size_t error_bound_bytes = error_bounds.data.size() * sizeof(float);
    std::cout << "[topoanns_pq2_residual_memory]"
              << " cross_term_bytes=" << cross_term_bytes
              << " error_bound_bytes=" << error_bound_bytes
              << " cuda_free_bytes_before_error_bound=" << free_bytes
              << " cuda_total_bytes=" << total_bytes << std::endl;

    aux->error_bounds_fp32 = CudaBuffer<float>::CopyFromHost(error_bounds.data);
    aux->error_bound_storage = Pq2ErrorBoundStorage::kFloat32;
    aux->error_bound_count = error_bounds.rows;
    pq2_aux_ = std::move(aux);
}

const PqIndex& SearchResources::pq2_index() const {
    if (!pq2_index_) {
        throw std::runtime_error(
            BuildErrorMessage("SearchResources::pq2_index", "PQ2 index is not loaded."));
    }
    return *pq2_index_;
}

bool SearchResources::pq2_is_residual_refine() const {
    return pq2_aux_ != nullptr && pq2_aux_->residual_refine;
}

bool SearchResources::has_pq2_error_bounds() const {
    return pq2_aux_ != nullptr &&
           pq2_aux_->error_bound_storage != Pq2ErrorBoundStorage::kNone;
}

Pq2ErrorBoundStorage SearchResources::pq2_error_bound_storage() const {
    if (!pq2_aux_) {
        return Pq2ErrorBoundStorage::kNone;
    }
    return pq2_aux_->error_bound_storage;
}

const CudaBuffer<float>& SearchResources::pq2_cross_terms() const {
    if (!pq2_aux_ || !pq2_aux_->residual_refine || pq2_aux_->cross_terms.empty()) {
        throw std::runtime_error(BuildErrorMessage("SearchResources::pq2_cross_terms",
                                                   "Residual PQ2 cross-term table is not loaded."));
    }
    return pq2_aux_->cross_terms;
}

const CudaBuffer<float>& SearchResources::pq2_error_bounds_fp32() const {
    if (!pq2_aux_ ||
        pq2_aux_->error_bound_storage != Pq2ErrorBoundStorage::kFloat32 ||
        pq2_aux_->error_bounds_fp32.empty()) {
        throw std::runtime_error(BuildErrorMessage(
            "SearchResources::pq2_error_bounds_fp32",
            "Residual PQ2 float32 error bounds are not loaded."));
    }
    return pq2_aux_->error_bounds_fp32;
}

void SearchResources::LoadHpqIndex(const std::filesystem::path& base_pivots_path,
                                   const std::filesystem::path& outlier_pivots_path,
                                   const std::filesystem::path& hybrid_codes_path,
                                   const std::filesystem::path& selector_bits_path) {
    if (!pq_index_) {
        throw std::runtime_error(BuildErrorMessage(
            "SearchResources::LoadHpqIndex", "Base PQ index must be loaded first."));
    }
    auto hpq = std::make_shared<HpqIndex>(
        HpqIndex::Load(base_pivots_path, outlier_pivots_path, hybrid_codes_path,
                       selector_bits_path));
    const auto& graph_base = pq_index_->host();
    const auto& base = hpq->base_index().host();
    const auto& outlier = hpq->outlier_index().host();
    if (base.num_points != outlier.num_points || base.ndims != outlier.ndims ||
        base.num_chunks != outlier.num_chunks ||
        base.chunk_offsets != outlier.chunk_offsets) {
        throw std::runtime_error(BuildErrorMessage(
            "SearchResources::LoadHpqIndex", "Base and outlier HPQ layouts differ."));
    }
    if (graph_base.num_points != base.num_points || graph_base.ndims != base.ndims) {
        throw std::runtime_error(BuildErrorMessage(
            "SearchResources::LoadHpqIndex", "Graph and HPQ dataset shapes differ."));
    }
    hpq_index_ = std::move(hpq);
}

const HpqIndex& SearchResources::hpq_index() const {
    if (!hpq_index_) {
        throw std::runtime_error(BuildErrorMessage(
            "SearchResources::hpq_index", "HPQ index is not loaded."));
    }
    return *hpq_index_;
}

void SearchResources::LoadVectorStore(const std::filesystem::path& path) {
    auto resource = std::make_shared<VectorStoreResource>();
    resource->path = path;
    resource->header = VectorStoreBuilder::ReadHeader(path);
    resource->layout = VectorPageLayout::CreateFromVectorBytes(
        static_cast<std::size_t>(resource->header.vector_bytes),
        static_cast<std::size_t>(resource->header.page_size_bytes));
    vector_store_ = std::move(resource);
}

const VectorStoreHeader& SearchResources::vector_store_header() const {
    if (!vector_store_) {
        throw std::runtime_error(BuildErrorMessage("SearchResources::vector_store_header",
                                                   "Vector store is not loaded."));
    }
    return vector_store_->header;
}

const VectorPageLayout& SearchResources::vector_store_layout() const {
    if (!vector_store_) {
        throw std::runtime_error(BuildErrorMessage("SearchResources::vector_store_layout",
                                                   "Vector store is not loaded."));
    }
    return vector_store_->layout;
}

const std::filesystem::path& SearchResources::vector_store_path() const {
    if (!vector_store_) {
        throw std::runtime_error(BuildErrorMessage("SearchResources::vector_store_path",
                                                   "Vector store is not loaded."));
    }
    return vector_store_->path;
}

void SearchResources::AttachVectorPageProvider(std::shared_ptr<VectorPageProvider> provider) {
    if (!vector_store_) {
        throw std::runtime_error(BuildErrorMessage("SearchResources::AttachVectorPageProvider",
                                                   "Vector store must be loaded first."));
    }
    if (!provider) {
        throw std::runtime_error(BuildErrorMessage("SearchResources::AttachVectorPageProvider",
                                                   "provider must not be null."));
    }
    vector_store_->provider = std::move(provider);
}

void SearchResources::ClearVectorPageProvider() {
    if (vector_store_) {
        vector_store_->provider.reset();
    }
}

const VectorPageProvider& SearchResources::vector_page_provider() const {
    if (!vector_store_ || !vector_store_->provider) {
        throw std::runtime_error(BuildErrorMessage("SearchResources::vector_page_provider",
                                                   "Vector page provider is not configured."));
    }
    return *vector_store_->provider;
}

void SearchResources::AttachTopologyPageProvider(std::shared_ptr<VectorPageProvider> provider) {
    if (!provider) {
        throw std::runtime_error(BuildErrorMessage("SearchResources::AttachTopologyPageProvider",
                                                   "provider must not be null."));
    }
    topology_page_provider_ = std::move(provider);
}

void SearchResources::ClearTopologyPageProvider() {
    topology_page_provider_.reset();
}

const VectorPageProvider& SearchResources::topology_page_provider() const {
    if (!topology_page_provider_) {
        throw std::runtime_error(BuildErrorMessage("SearchResources::topology_page_provider",
                                                   "Topology page provider is not configured."));
    }
    return *topology_page_provider_;
}

const void* SearchResources::topology_device_read_handle() const {
    return topology_page_provider_ ? topology_page_provider_->DeviceReadHandle() : nullptr;
}

void SearchResources::AttachCombinedNodePageProvider(
    std::shared_ptr<VectorPageProvider> provider) {
    if (!provider) {
        throw std::runtime_error(BuildErrorMessage(
            "SearchResources::AttachCombinedNodePageProvider",
            "provider must not be null."));
    }
    combined_node_page_provider_ = std::move(provider);
}

void SearchResources::ClearCombinedNodePageProvider() {
    combined_node_page_provider_.reset();
}

const void* SearchResources::combined_node_device_read_handle() const {
    return combined_node_page_provider_
               ? combined_node_page_provider_->DeviceReadHandle()
               : nullptr;
}

}  // namespace topoanns
