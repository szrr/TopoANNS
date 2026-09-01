#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "topoanns/pinned_topology.hpp"
#include "topoanns/hpq_index.hpp"
#include "topoanns/pq_index.hpp"
#include "topoanns/topology_layout.hpp"
#include "topoanns/vector_page_provider.hpp"
#include "topoanns/vector_page_layout.hpp"
#include "topoanns/vector_store_builder.hpp"

namespace topoanns {

enum class Pq2ErrorBoundStorage {
    kNone = 0,
    kFloat32 = 1,
};

class SearchResources {
public:
    SearchResources() = default;

    static SearchResources FromDataset(const TopologyDataset& dataset);
    static SearchResources FromTopologyFile(const std::filesystem::path& path);

    std::uint64_t num_nodes() const;
    std::uint32_t degree() const;
    const PinnedTopology& topology() const { return topology_; }
    const std::uint32_t* host_topology_data() const { return topology_.host_data(); }
    const std::uint32_t* device_topology_data() const { return topology_.device_data(); }

    std::vector<std::uint32_t> ReadHostNeighbors(std::uint64_t node_id) const;
    void LoadPqIndex(const std::filesystem::path& pivots_path,
                     const std::filesystem::path& compressed_path);
    bool has_pq_index() const { return static_cast<bool>(pq_index_); }
    const PqIndex& pq_index() const;
    void LoadPq2Index(const std::filesystem::path& pivots_path,
                      const std::filesystem::path& compressed_path,
                      const std::filesystem::path& error_bounds_path = {});
    bool has_pq2_index() const { return static_cast<bool>(pq2_index_); }
    const PqIndex& pq2_index() const;
    bool pq2_is_residual_refine() const;
    bool has_pq2_error_bounds() const;
    Pq2ErrorBoundStorage pq2_error_bound_storage() const;
    const CudaBuffer<float>& pq2_cross_terms() const;
    const CudaBuffer<float>& pq2_error_bounds_fp32() const;
    void LoadHpqIndex(const std::filesystem::path& base_pivots_path,
                      const std::filesystem::path& outlier_pivots_path,
                      const std::filesystem::path& hybrid_codes_path,
                      const std::filesystem::path& selector_bits_path);
    bool has_hpq_index() const { return static_cast<bool>(hpq_index_); }
    const HpqIndex& hpq_index() const;

    void LoadVectorStore(const std::filesystem::path& path);
    bool has_vector_store() const { return static_cast<bool>(vector_store_); }
    const VectorStoreHeader& vector_store_header() const;
    const VectorPageLayout& vector_store_layout() const;
    const std::filesystem::path& vector_store_path() const;
    void AttachVectorPageProvider(std::shared_ptr<VectorPageProvider> provider);
    void ClearVectorPageProvider();
    const VectorPageProvider& vector_page_provider() const;
    void AttachTopologyPageProvider(std::shared_ptr<VectorPageProvider> provider);
    void ClearTopologyPageProvider();
    bool has_topology_page_provider() const { return static_cast<bool>(topology_page_provider_); }
    const VectorPageProvider& topology_page_provider() const;
    const void* topology_device_read_handle() const;
    void AttachCombinedNodePageProvider(std::shared_ptr<VectorPageProvider> provider);
    void ClearCombinedNodePageProvider();
    bool has_combined_node_page_provider() const {
        return static_cast<bool>(combined_node_page_provider_);
    }
    const void* combined_node_device_read_handle() const;

private:
    PinnedTopology topology_;
    std::shared_ptr<VectorPageProvider> topology_page_provider_;
    std::shared_ptr<VectorPageProvider> combined_node_page_provider_;
    std::shared_ptr<PqIndex> pq_index_;
    std::shared_ptr<PqIndex> pq2_index_;
    struct Pq2AuxResource {
        bool residual_refine = false;
        Pq2ErrorBoundStorage error_bound_storage = Pq2ErrorBoundStorage::kNone;
        std::uint64_t error_bound_count = 0;
        CudaBuffer<float> cross_terms;
        CudaBuffer<float> error_bounds_fp32;
    };
    std::shared_ptr<Pq2AuxResource> pq2_aux_;
    std::shared_ptr<HpqIndex> hpq_index_;
    struct VectorStoreResource {
        std::filesystem::path path;
        VectorStoreHeader header{};
        VectorPageLayout layout;
        std::shared_ptr<VectorPageProvider> provider;
    };
    std::shared_ptr<VectorStoreResource> vector_store_;
};

}  // namespace topoanns
