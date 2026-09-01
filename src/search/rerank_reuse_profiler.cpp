#include "rerank_reuse_profiler.hpp"

#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace topoanns::detail {
namespace {

constexpr const char* kProfileRatiosEnv = "TOPOANNS_RERANK_REUSE_PROFILE_RATIOS";

std::vector<double> ParseRatios() {
    const char* value = std::getenv(kProfileRatiosEnv);
    if (value == nullptr || value[0] == '\0') {
        return {};
    }

    std::vector<double> ratios;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (token.empty()) {
            continue;
        }
        const double ratio = std::stod(token);
        if (ratio < 0.0 || ratio > 1.0) {
            throw std::runtime_error(
                std::string(kProfileRatiosEnv) + " values must be in [0, 1].");
        }
        ratios.push_back(ratio);
    }
    std::sort(ratios.begin(), ratios.end());
    ratios.erase(std::unique(ratios.begin(), ratios.end()), ratios.end());
    if (ratios.empty()) {
        throw std::runtime_error(std::string(kProfileRatiosEnv) + " is empty.");
    }
    return ratios;
}

const std::vector<double>& ProfileRatios() {
    static const std::vector<double> ratios = ParseRatios();
    return ratios;
}

struct ProfileAccumulator {
    std::uint64_t num_nodes = 0;
    std::uint64_t queries = 0;
    std::vector<std::uint64_t> valid_by_rank;
    std::vector<std::uint64_t> expanded_by_rank;
    std::vector<std::vector<std::uint64_t>> reusable_by_ratio_rank;
};

std::mutex& ProfileMutex() {
    static std::mutex mutex;
    return mutex;
}

std::map<std::size_t, ProfileAccumulator>& Accumulators() {
    static std::map<std::size_t, ProfileAccumulator> accumulators;
    return accumulators;
}

}  // namespace

bool RerankReuseProfilingEnabled() {
    return !ProfileRatios().empty();
}

void ProfileRerankReuseCandidates(
    const CudaBuffer<DeviceTopologyCandidate>& traversal_candidate_buffer,
    std::size_t traversal_candidate_capacity,
    const CudaBuffer<DeviceTopologyCandidate>& rerank_candidate_buffer,
    std::size_t num_queries,
    std::size_t rerank_candidate_capacity,
    std::size_t top_n,
    std::uint64_t num_nodes) {
    const std::vector<double>& ratios = ProfileRatios();
    if (ratios.empty() || num_queries == 0 || top_n == 0) {
        return;
    }
    if (top_n > rerank_candidate_capacity ||
        traversal_candidate_buffer.size() < num_queries * traversal_candidate_capacity ||
        rerank_candidate_buffer.size() < num_queries * rerank_candidate_capacity) {
        throw std::runtime_error("Invalid candidate buffer shape for rerank reuse profiling.");
    }

    const std::vector<DeviceTopologyCandidate> traversal_candidates =
        traversal_candidate_buffer.CopyToHost();
    const std::vector<DeviceTopologyCandidate> rerank_candidates =
        rerank_candidate_buffer.CopyToHost();
    std::vector<std::uint64_t> valid_by_rank(top_n, 0);
    std::vector<std::uint64_t> expanded_by_rank(top_n, 0);
    std::vector<std::vector<std::uint64_t>> reusable_by_ratio_rank(
        ratios.size(), std::vector<std::uint64_t>(top_n, 0));
    std::vector<std::uint64_t> cached_node_counts;
    cached_node_counts.reserve(ratios.size());
    for (const double ratio : ratios) {
        cached_node_counts.push_back(std::min<std::uint64_t>(
            num_nodes, static_cast<std::uint64_t>(ratio * static_cast<double>(num_nodes))));
    }

    for (std::size_t query_id = 0; query_id < num_queries; ++query_id) {
        const std::size_t traversal_base = query_id * traversal_candidate_capacity;
        const std::size_t rerank_base = query_id * rerank_candidate_capacity;
        for (std::size_t rank = 0; rank < top_n; ++rank) {
            const DeviceTopologyCandidate& candidate = rerank_candidates[rerank_base + rank];
            if (!candidate.valid()) {
                continue;
            }
            ++valid_by_rank[rank];
            const std::uint32_t node_id = candidate.raw_node_id();
            bool was_expanded = candidate.expanded();
            for (std::size_t traversal_rank = 0;
                 !was_expanded && traversal_rank < traversal_candidate_capacity;
                 ++traversal_rank) {
                const DeviceTopologyCandidate& traversal_candidate =
                    traversal_candidates[traversal_base + traversal_rank];
                was_expanded = traversal_candidate.valid() &&
                               traversal_candidate.raw_node_id() == node_id &&
                               traversal_candidate.expanded();
            }
            if (!was_expanded) {
                continue;
            }
            ++expanded_by_rank[rank];
            for (std::size_t ratio_idx = 0; ratio_idx < ratios.size(); ++ratio_idx) {
                if (node_id >= cached_node_counts[ratio_idx]) {
                    ++reusable_by_ratio_rank[ratio_idx][rank];
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(ProfileMutex());
    ProfileAccumulator& accumulator = Accumulators()[top_n];
    if (accumulator.valid_by_rank.empty()) {
        accumulator.num_nodes = num_nodes;
        accumulator.valid_by_rank.assign(top_n, 0);
        accumulator.expanded_by_rank.assign(top_n, 0);
        accumulator.reusable_by_ratio_rank.assign(
            ratios.size(), std::vector<std::uint64_t>(top_n, 0));
    } else if (accumulator.num_nodes != num_nodes) {
        throw std::runtime_error("Rerank reuse profiler cannot mix datasets.");
    }
    accumulator.queries += num_queries;
    for (std::size_t rank = 0; rank < top_n; ++rank) {
        accumulator.valid_by_rank[rank] += valid_by_rank[rank];
        accumulator.expanded_by_rank[rank] += expanded_by_rank[rank];
        for (std::size_t ratio_idx = 0; ratio_idx < ratios.size(); ++ratio_idx) {
            accumulator.reusable_by_ratio_rank[ratio_idx][rank] +=
                reusable_by_ratio_rank[ratio_idx][rank];
        }
    }
}

void PrintRerankReuseProfileSummary() {
    const std::vector<double>& ratios = ProfileRatios();
    if (ratios.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(ProfileMutex());
    for (const auto& [top_n, accumulator] : Accumulators()) {
        for (std::size_t ratio_idx = 0; ratio_idx < ratios.size(); ++ratio_idx) {
            std::uint64_t cumulative_valid = 0;
            std::uint64_t cumulative_expanded = 0;
            std::uint64_t cumulative_reusable = 0;
            const std::uint64_t cached_node_count = std::min<std::uint64_t>(
                accumulator.num_nodes,
                static_cast<std::uint64_t>(
                    ratios[ratio_idx] * static_cast<double>(accumulator.num_nodes)));
            for (std::size_t rank = 0; rank < top_n; ++rank) {
                cumulative_valid += accumulator.valid_by_rank[rank];
                cumulative_expanded += accumulator.expanded_by_rank[rank];
                cumulative_reusable +=
                    accumulator.reusable_by_ratio_rank[ratio_idx][rank];
                const std::size_t prefix = rank + 1;
                const bool report_prefix =
                    prefix == top_n || prefix <= 32 || (prefix & (prefix - 1)) == 0;
                if (!report_prefix) {
                    continue;
                }
                const double reusable_per_query =
                    accumulator.queries == 0
                        ? 0.0
                        : static_cast<double>(cumulative_reusable) /
                              static_cast<double>(accumulator.queries);
                const double reusable_frac_valid =
                    cumulative_valid == 0
                        ? 0.0
                        : static_cast<double>(cumulative_reusable) /
                              static_cast<double>(cumulative_valid);
                const double reusable_frac_expanded =
                    cumulative_expanded == 0
                        ? 0.0
                        : static_cast<double>(cumulative_reusable) /
                              static_cast<double>(cumulative_expanded);
                std::cout << std::setprecision(10)
                          << "[topoanns_rerank_reuse_profile]"
                          << " top_n=" << top_n
                          << " prefix=" << prefix
                          << " cache_ratio=" << ratios[ratio_idx]
                          << " cached_node_count=" << cached_node_count
                          << " queries=" << accumulator.queries
                          << " valid_candidates=" << cumulative_valid
                          << " expanded_candidates=" << cumulative_expanded
                          << " reusable_candidates=" << cumulative_reusable
                          << " reusable_per_query=" << reusable_per_query
                          << " reusable_frac_valid=" << reusable_frac_valid
                          << " reusable_frac_expanded=" << reusable_frac_expanded
                          << '\n';
            }
        }
    }
}

}  // namespace topoanns::detail
