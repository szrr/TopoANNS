#pragma once

#include <cstddef>
#include <cstdint>

#include "topology_search_kernel.hpp"

namespace topoanns::detail {

bool RerankReuseProfilingEnabled();

void ProfileRerankReuseCandidates(
    const CudaBuffer<DeviceTopologyCandidate>& traversal_candidate_buffer,
    std::size_t traversal_candidate_capacity,
    const CudaBuffer<DeviceTopologyCandidate>& rerank_candidate_buffer,
    std::size_t num_queries,
    std::size_t rerank_candidate_capacity,
    std::size_t top_n,
    std::uint64_t num_nodes);

void PrintRerankReuseProfileSummary();

}  // namespace topoanns::detail
