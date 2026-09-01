#include "topoanns/rvq_entry_provider.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/pq_distance_oracle.hpp"
#include "../src/search/topology_search_kernel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::filesystem::path index_dir;
    std::filesystem::path rvq_model;
    std::filesystem::path query_bin;
    std::filesystem::path gt_bin;
    std::filesystem::path pq2_pivots;
    std::filesystem::path pq2_codes;
    std::size_t num_queries = 0;
    std::size_t batch_size = 256;
    std::size_t top_k = 10;
    std::vector<std::size_t> top_l_values;
    std::size_t search_width = 2;
    std::size_t max_expansions = 4096;
    std::size_t rvq_entry_count = 128;
    std::vector<std::size_t> overlap_thresholds{50, 100, 200, 500, 1000};
};

struct FloatMatrix {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<float> values;
};

struct IntMatrix {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<std::int32_t> values;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_overlap_eval"
        << " --index-dir <path>"
        << " --rvq-model <path>"
        << " --query-bin <path>"
        << " --gt-bin <path>"
        << " --num-queries <count>"
        << " --top-l-values <csv>"
        << " [--pq2-pivots <path> --pq2-codes <path>]"
        << " [--batch-size <count>]"
        << " [--top-k <count>]"
        << " [--search-width <count>]"
        << " [--max-expansions <count>]"
        << " [--rvq-entry-count <count>]"
        << " [--overlap-thresholds <csv>]"
        << std::endl;
    std::exit(1);
}

std::vector<std::size_t> ParseCsvSizes(const std::string& text) {
    std::vector<std::size_t> values;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token =
            text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (!token.empty()) {
            values.push_back(std::stoull(token));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    if (values.empty()) {
        throw std::runtime_error("CSV list must not be empty.");
    }
    return values;
}

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        auto read_value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (flag == "--index-dir") {
            args.index_dir = read_value("--index-dir");
        } else if (flag == "--rvq-model") {
            args.rvq_model = read_value("--rvq-model");
        } else if (flag == "--query-bin") {
            args.query_bin = read_value("--query-bin");
        } else if (flag == "--gt-bin") {
            args.gt_bin = read_value("--gt-bin");
        } else if (flag == "--pq2-pivots") {
            args.pq2_pivots = read_value("--pq2-pivots");
        } else if (flag == "--pq2-codes") {
            args.pq2_codes = read_value("--pq2-codes");
        } else if (flag == "--num-queries") {
            args.num_queries = std::stoull(read_value("--num-queries"));
        } else if (flag == "--batch-size") {
            args.batch_size = std::stoull(read_value("--batch-size"));
        } else if (flag == "--top-k") {
            args.top_k = std::stoull(read_value("--top-k"));
        } else if (flag == "--top-l-values") {
            args.top_l_values = ParseCsvSizes(read_value("--top-l-values"));
        } else if (flag == "--search-width") {
            args.search_width = std::stoull(read_value("--search-width"));
        } else if (flag == "--max-expansions") {
            args.max_expansions = std::stoull(read_value("--max-expansions"));
        } else if (flag == "--rvq-entry-count") {
            args.rvq_entry_count = std::stoull(read_value("--rvq-entry-count"));
        } else if (flag == "--overlap-thresholds") {
            args.overlap_thresholds = ParseCsvSizes(read_value("--overlap-thresholds"));
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }

    if (args.index_dir.empty() || args.rvq_model.empty() || args.query_bin.empty() ||
        args.gt_bin.empty() || args.num_queries == 0 || args.top_l_values.empty()) {
        Usage();
    }
    if (args.pq2_pivots.empty() != args.pq2_codes.empty()) {
        throw std::runtime_error("PQ2 pivots/codes must be provided together.");
    }
    std::sort(args.top_l_values.begin(), args.top_l_values.end());
    args.top_l_values.erase(std::unique(args.top_l_values.begin(), args.top_l_values.end()),
                            args.top_l_values.end());
    std::sort(args.overlap_thresholds.begin(), args.overlap_thresholds.end());
    args.overlap_thresholds.erase(
        std::unique(args.overlap_thresholds.begin(), args.overlap_thresholds.end()),
        args.overlap_thresholds.end());
    return args;
}

FloatMatrix LoadFloatMatrix(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open float matrix: " + path.string());
    }
    FloatMatrix matrix;
    in.read(reinterpret_cast<char*>(&matrix.rows), sizeof(matrix.rows));
    in.read(reinterpret_cast<char*>(&matrix.cols), sizeof(matrix.cols));
    if (!in.good() || matrix.rows == 0 || matrix.cols == 0) {
        throw std::runtime_error("Invalid float matrix header: " + path.string());
    }
    matrix.values.resize(static_cast<std::size_t>(matrix.rows) * matrix.cols);
    in.read(reinterpret_cast<char*>(matrix.values.data()),
            static_cast<std::streamsize>(matrix.values.size() * sizeof(float)));
    if (in.gcount() != static_cast<std::streamsize>(matrix.values.size() * sizeof(float))) {
        throw std::runtime_error("Short read in float matrix: " + path.string());
    }
    return matrix;
}

IntMatrix LoadIntMatrix(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open int matrix: " + path.string());
    }
    IntMatrix matrix;
    in.read(reinterpret_cast<char*>(&matrix.rows), sizeof(matrix.rows));
    in.read(reinterpret_cast<char*>(&matrix.cols), sizeof(matrix.cols));
    if (!in.good() || matrix.rows == 0 || matrix.cols == 0) {
        throw std::runtime_error("Invalid int matrix header: " + path.string());
    }
    const std::size_t total = static_cast<std::size_t>(matrix.rows) * matrix.cols;
    matrix.values.resize(total);
    in.read(reinterpret_cast<char*>(matrix.values.data()),
            static_cast<std::streamsize>(total * sizeof(std::int32_t)));
    if (in.gcount() != static_cast<std::streamsize>(total * sizeof(std::int32_t))) {
        throw std::runtime_error("Short read in int matrix: " + path.string());
    }
    return matrix;
}

struct OverlapTotals {
    std::size_t threshold = 0;
    std::size_t matched = 0;
};

struct ScoredNode {
    float distance = 0.0f;
    std::uint32_t node_id = topoanns::kInvalidNodeId;
};

std::size_t Popcount(std::uint32_t value) {
    return static_cast<std::size_t>(__builtin_popcount(value));
}

bool ScoredNodeLess(const ScoredNode& lhs, const ScoredNode& rhs) {
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    return lhs.node_id < rhs.node_id;
}

std::vector<std::uint32_t> BuildSearchPrefix(
    const topoanns::detail::DeviceTopologyCandidate* candidates,
    std::size_t candidate_count,
    std::size_t prefix_limit) {
    const std::size_t kept = std::min(candidate_count, prefix_limit);
    std::vector<std::uint32_t> ranked_nodes;
    ranked_nodes.reserve(kept);
    for (std::size_t rank = 0; rank < kept; ++rank) {
        const auto& candidate = candidates[rank];
        if (!candidate.valid()) {
            break;
        }
        ranked_nodes.push_back(candidate.raw_node_id());
    }
    return ranked_nodes;
}

std::vector<std::uint32_t> BuildPqRerankedPrefix(
    const topoanns::detail::DeviceTopologyCandidate* candidates,
    std::size_t candidate_count,
    std::size_t prefix_limit,
    std::size_t query_id,
    const topoanns::PqDistanceOracle& oracle) {
    std::vector<ScoredNode> scored_nodes;
    scored_nodes.reserve(candidate_count);
    for (std::size_t rank = 0; rank < candidate_count; ++rank) {
        const auto& candidate = candidates[rank];
        if (!candidate.valid()) {
            continue;
        }
        const std::uint32_t raw_node_id = candidate.raw_node_id();
        scored_nodes.push_back(
            ScoredNode{oracle.Distance(query_id, raw_node_id), raw_node_id});
    }

    const std::size_t kept = std::min(prefix_limit, scored_nodes.size());
    if (kept == 0) {
        return {};
    }
    std::partial_sort(scored_nodes.begin(), scored_nodes.begin() + kept, scored_nodes.end(),
                      ScoredNodeLess);

    std::vector<std::uint32_t> ranked_nodes;
    ranked_nodes.reserve(kept);
    for (std::size_t i = 0; i < kept; ++i) {
        ranked_nodes.push_back(scored_nodes[i].node_id);
    }
    return ranked_nodes;
}

void AccumulateOverlapTotals(const std::vector<std::uint32_t>& ranked_nodes,
                             const std::uint32_t* gt_row,
                             std::size_t top_k,
                             std::vector<OverlapTotals>* totals) {
    for (auto& total : *totals) {
        const std::size_t candidate_limit = std::min(total.threshold, ranked_nodes.size());
        for (std::size_t gt_rank = 0; gt_rank < top_k; ++gt_rank) {
            const std::uint32_t target = gt_row[gt_rank];
            bool found = false;
            for (std::size_t candidate_rank = 0; candidate_rank < candidate_limit;
                 ++candidate_rank) {
                if (ranked_nodes[candidate_rank] == target) {
                    found = true;
                    break;
                }
            }
            total.matched += found ? 1U : 0U;
        }
    }
}

int Main(const Args& args) {
    const auto query_load_begin = std::chrono::steady_clock::now();
    const FloatMatrix queries = LoadFloatMatrix(args.query_bin);
    const IntMatrix gt = LoadIntMatrix(args.gt_bin);
    const auto query_load_end = std::chrono::steady_clock::now();

    if (args.num_queries > queries.rows || args.num_queries > gt.rows) {
        throw std::runtime_error("num-queries exceeds query/gt rows.");
    }
    if (queries.cols == 0 || gt.cols < args.top_k) {
        throw std::runtime_error("Invalid query or GT dimensions.");
    }

    const auto topology_load_begin = std::chrono::steady_clock::now();
    topoanns::SearchResources resources = topoanns::SearchResources::FromTopologyFile(
        args.index_dir / "topology.bin");
    const auto topology_load_end = std::chrono::steady_clock::now();

    const auto pq_load_begin = std::chrono::steady_clock::now();
    resources.LoadPqIndex(args.index_dir / "_pq_pivots.bin",
                          args.index_dir / "_pq_compressed.bin");
    const auto pq_load_end = std::chrono::steady_clock::now();

    double init_pq2_ms = 0.0;
    if (!args.pq2_pivots.empty()) {
        const auto pq2_load_begin = std::chrono::steady_clock::now();
        resources.LoadPq2Index(args.pq2_pivots, args.pq2_codes);
        const auto pq2_load_end = std::chrono::steady_clock::now();
        init_pq2_ms =
            std::chrono::duration<double, std::milli>(pq2_load_end - pq2_load_begin).count();
        if (!resources.pq2_index().host().codes_on_host) {
            throw std::runtime_error(
                "PQ2 overlap eval currently requires PQ2 codes retained on host. "
                "Use a smaller PQ2 index for validation before adding device-side rerank.");
        }
    }

    const auto rvq_load_begin = std::chrono::steady_clock::now();
    topoanns::RvqModel rvq_model = topoanns::RvqModel::Load(args.rvq_model);
    rvq_model.WarmUp();
    const auto rvq_load_end = std::chrono::steady_clock::now();

    cudaDeviceProp props{};
    if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties failed.");
    }
    const double cycles_per_ms = static_cast<double>(props.clockRate);

    const double init_query_ms =
        std::chrono::duration<double, std::milli>(query_load_end - query_load_begin).count();
    const double init_topology_ms =
        std::chrono::duration<double, std::milli>(topology_load_end - topology_load_begin).count();
    const double init_pq_ms =
        std::chrono::duration<double, std::milli>(pq_load_end - pq_load_begin).count();
    const double init_rvq_ms =
        std::chrono::duration<double, std::milli>(rvq_load_end - rvq_load_begin).count();
    std::cout << "[topoanns_overlap_init] query_load_ms=" << init_query_ms
              << " topology_load_ms=" << init_topology_ms
              << " pq_load_ms=" << init_pq_ms
              << " pq2_load_ms=" << init_pq2_ms
              << " rvq_load_ms=" << init_rvq_ms
              << std::endl;
    for (const std::size_t top_l : args.top_l_values) {
        std::vector<OverlapTotals> search_totals;
        search_totals.reserve(args.overlap_thresholds.size());
        for (const std::size_t threshold : args.overlap_thresholds) {
            search_totals.push_back(OverlapTotals{threshold, 0});
        }
        std::vector<OverlapTotals> pq2_totals = search_totals;

        topoanns::RvqEntryProfile aggregate_rvq_profile;
        double aggregate_topology_kernel_ms = 0.0;
        double aggregate_candidate_download_ms = 0.0;
        double aggregate_gt_mask_download_ms = 0.0;
        double aggregate_stats_download_ms = 0.0;
        double aggregate_profile_download_ms = 0.0;
        double aggregate_pq2_tables_ms = 0.0;
        double aggregate_query_pq_ms = 0.0;
        double aggregate_query_pq_compute_ms = 0.0;
        double aggregate_query_pq_prefetch_issue_ms = 0.0;
        double aggregate_query_pq_prefetch_wait_ms = 0.0;
        double aggregate_query_pq_checksum_ms = 0.0;
        double aggregate_query_queue_ms = 0.0;
        double aggregate_query_queue_scan_ms = 0.0;
        double aggregate_query_queue_select_ms = 0.0;
        double aggregate_query_frontier_sort_ms = 0.0;
        double aggregate_query_tail_merge_ms = 0.0;
        double aggregate_query_candidate_sort_ms = 0.0;
        double aggregate_query_hash_rebuild_ms = 0.0;
        std::size_t total_visited = 0;
        std::size_t total_expanded = 0;
        std::size_t total_gt_visited = 0;
        const std::size_t max_overlap_threshold = args.overlap_thresholds.back();

        const auto search_begin = std::chrono::steady_clock::now();
        for (std::size_t query_offset = 0; query_offset < args.num_queries;
             query_offset += args.batch_size) {
            const std::size_t batch_queries =
                std::min(args.batch_size, args.num_queries - query_offset);
            const float* begin =
                queries.values.data() +
                query_offset * static_cast<std::size_t>(queries.cols);
            const float* end = begin + batch_queries * static_cast<std::size_t>(queries.cols);
            std::vector<float> batch_query_buffer(begin, end);

            topoanns::RvqEntryProfile entry_profile;
            topoanns::DeviceEntryBatch entry_batch =
                rvq_model.ComputeFloat32DeviceEntryBatch(batch_query_buffer, batch_queries,
                                                         args.rvq_entry_count, 0,
                                                         &entry_profile);

            const topoanns::PqDistanceOracle oracle =
                topoanns::PqDistanceOracle::FromFloatQueries(resources, batch_query_buffer,
                                                             batch_queries);
            std::optional<topoanns::PqDistanceOracle> pq2_oracle;
            if (resources.has_pq2_index()) {
                pq2_oracle.emplace(topoanns::PqDistanceOracle::FromFloatQueries(
                    resources.pq2_index(), batch_query_buffer, batch_queries));
                aggregate_pq2_tables_ms += pq2_oracle->query_tables().profile().total_ms;
            }
            topoanns::TopologySearchParams params;
            params.top_k = std::max(args.top_k, args.overlap_thresholds.back());
            params.top_l = top_l;
            params.candidate_queue_size = std::max(top_l, params.top_k);
            params.search_width = args.search_width;
            params.max_expansions = args.max_expansions;

            std::vector<std::uint32_t> host_gt_ids(batch_queries * args.top_k,
                                                   topoanns::kInvalidNodeId);
            for (std::size_t batch_query_id = 0; batch_query_id < batch_queries;
                 ++batch_query_id) {
                const std::int32_t* gt_row =
                    gt.values.data() +
                    (query_offset + batch_query_id) * static_cast<std::size_t>(gt.cols);
                for (std::size_t gt_rank = 0; gt_rank < args.top_k; ++gt_rank) {
                    host_gt_ids[batch_query_id * args.top_k + gt_rank] =
                        static_cast<std::uint32_t>(gt_row[gt_rank]);
                }
            }
            topoanns::CudaBuffer<std::uint32_t> gt_ids_buffer =
                topoanns::CudaBuffer<std::uint32_t>::CopyFromHost(host_gt_ids);
            topoanns::detail::DeviceTopologyBatchResult device_result =
                topoanns::detail::RunTopologySearchKernelBatchDevice(
                    resources, oracle, entry_batch, gt_ids_buffer,
                    static_cast<std::uint32_t>(args.top_k), params);

            const auto candidate_download_begin = std::chrono::steady_clock::now();
            const std::vector<topoanns::detail::DeviceTopologyCandidate> host_candidates =
                device_result.candidate_buffer.CopyToHost();
            const auto candidate_download_end = std::chrono::steady_clock::now();
            const auto gt_mask_download_begin = std::chrono::steady_clock::now();
            const std::vector<std::uint32_t> host_gt_hit_masks =
                device_result.gt_hit_mask_buffer.CopyToHost();
            const auto gt_mask_download_end = std::chrono::steady_clock::now();
            const auto stats_download_begin = std::chrono::steady_clock::now();
            const std::vector<topoanns::detail::DeviceTopologySearchStats> host_stats =
                device_result.stats_buffer.CopyToHost();
            const auto stats_download_end = std::chrono::steady_clock::now();
            const auto profile_download_begin = std::chrono::steady_clock::now();
            const std::vector<topoanns::detail::DeviceTopologyProfileCycles> host_profiles =
                device_result.profile_buffer.CopyToHost();
            const auto profile_download_end = std::chrono::steady_clock::now();

            aggregate_rvq_profile.total_ms += entry_profile.total_ms;
            aggregate_rvq_profile.query_upload_ms += entry_profile.query_upload_ms;
            aggregate_rvq_profile.search_kernel_ms += entry_profile.search_kernel_ms;
            aggregate_rvq_profile.entry_gather_ms += entry_profile.entry_gather_ms;
            aggregate_topology_kernel_ms += device_result.kernel_ms;
            aggregate_candidate_download_ms +=
                std::chrono::duration<double, std::milli>(candidate_download_end -
                                                          candidate_download_begin)
                    .count();
            aggregate_gt_mask_download_ms +=
                std::chrono::duration<double, std::milli>(gt_mask_download_end -
                                                          gt_mask_download_begin)
                    .count();
            aggregate_stats_download_ms +=
                std::chrono::duration<double, std::milli>(stats_download_end -
                                                          stats_download_begin)
                    .count();
            aggregate_profile_download_ms +=
                std::chrono::duration<double, std::milli>(profile_download_end -
                                                          profile_download_begin)
                    .count();

            for (std::size_t batch_query_id = 0; batch_query_id < batch_queries;
                 ++batch_query_id) {
                const auto& query_profile = host_profiles[batch_query_id];
                aggregate_query_pq_ms +=
                    static_cast<double>(query_profile.pq_cycles) / cycles_per_ms;
                aggregate_query_pq_compute_ms +=
                    static_cast<double>(query_profile.pq_compute_cycles) / cycles_per_ms;
                aggregate_query_pq_prefetch_issue_ms +=
                    static_cast<double>(query_profile.pq_prefetch_issue_cycles) / cycles_per_ms;
                aggregate_query_pq_prefetch_wait_ms +=
                    static_cast<double>(query_profile.pq_prefetch_wait_cycles) / cycles_per_ms;
                aggregate_query_pq_checksum_ms +=
                    static_cast<double>(query_profile.pq_checksum_cycles) / cycles_per_ms;
                aggregate_query_queue_ms +=
                    static_cast<double>(query_profile.queue_cycles) / cycles_per_ms;
                aggregate_query_queue_scan_ms +=
                    static_cast<double>(query_profile.queue_scan_cycles) / cycles_per_ms;
                aggregate_query_queue_select_ms +=
                    static_cast<double>(query_profile.queue_select_cycles) / cycles_per_ms;
                aggregate_query_frontier_sort_ms +=
                    static_cast<double>(query_profile.frontier_sort_cycles) / cycles_per_ms;
                aggregate_query_tail_merge_ms +=
                    static_cast<double>(query_profile.tail_merge_cycles) / cycles_per_ms;
                aggregate_query_candidate_sort_ms +=
                    static_cast<double>(query_profile.candidate_sort_cycles) / cycles_per_ms;
                aggregate_query_hash_rebuild_ms +=
                    static_cast<double>(query_profile.hash_rebuild_cycles) / cycles_per_ms;
                total_visited += host_stats[batch_query_id].visited_nodes;
                total_expanded += host_stats[batch_query_id].expanded_nodes;
                total_gt_visited += Popcount(host_gt_hit_masks[batch_query_id]);
                const auto* query_candidates =
                    host_candidates.data() + batch_query_id * device_result.candidate_capacity;
                const std::size_t valid_candidates =
                    std::min<std::size_t>(host_stats[batch_query_id].valid_candidates,
                                          device_result.candidate_capacity);
                const std::uint32_t* gt_row =
                    host_gt_ids.data() + batch_query_id * args.top_k;

                const std::vector<std::uint32_t> search_ranked_nodes =
                    BuildSearchPrefix(query_candidates, valid_candidates, max_overlap_threshold);
                AccumulateOverlapTotals(search_ranked_nodes, gt_row, args.top_k, &search_totals);

                if (pq2_oracle.has_value()) {
                    const std::vector<std::uint32_t> pq2_ranked_nodes =
                        BuildPqRerankedPrefix(query_candidates, valid_candidates,
                                              max_overlap_threshold, batch_query_id,
                                              *pq2_oracle);
                    AccumulateOverlapTotals(pq2_ranked_nodes, gt_row, args.top_k, &pq2_totals);
                }
            }

            std::cout << "[topoanns_overlap_progress] top_l=" << top_l
                      << " finished queries " << (query_offset + batch_queries)
                      << " / " << args.num_queries << std::endl;
        }
        const auto search_end = std::chrono::steady_clock::now();
        const double search_ms =
            std::chrono::duration<double, std::milli>(search_end - search_begin).count();

        std::cout << "[topoanns_overlap_profile] top_l=" << top_l
                  << " rvq_total_ms=" << aggregate_rvq_profile.total_ms
                  << " rvq_query_upload_ms=" << aggregate_rvq_profile.query_upload_ms
                  << " rvq_kernel_ms=" << aggregate_rvq_profile.search_kernel_ms
                  << " rvq_entry_gather_ms=" << aggregate_rvq_profile.entry_gather_ms
                  << " topology_kernel_ms=" << aggregate_topology_kernel_ms
                  << " topology_candidate_download_ms=" << aggregate_candidate_download_ms
                  << " topology_gt_mask_download_ms=" << aggregate_gt_mask_download_ms
                  << " topology_stats_download_ms=" << aggregate_stats_download_ms
                  << " topology_profile_download_ms=" << aggregate_profile_download_ms
                  << " pq2_query_tables_ms=" << aggregate_pq2_tables_ms
                  << " search_wall_ms=" << search_ms
                  << " avg_query_pq_ms=" << (aggregate_query_pq_ms / args.num_queries)
                  << " avg_query_pq_compute_ms="
                  << (aggregate_query_pq_compute_ms / args.num_queries)
                  << " avg_query_pq_prefetch_issue_ms="
                  << (aggregate_query_pq_prefetch_issue_ms / args.num_queries)
                  << " avg_query_pq_prefetch_wait_ms="
                  << (aggregate_query_pq_prefetch_wait_ms / args.num_queries)
                  << " avg_query_pq_checksum_ms="
                  << (aggregate_query_pq_checksum_ms / args.num_queries)
                  << " avg_query_queue_ms=" << (aggregate_query_queue_ms / args.num_queries)
                  << " avg_query_queue_scan_ms="
                  << (aggregate_query_queue_scan_ms / args.num_queries)
                  << " avg_query_queue_select_ms="
                  << (aggregate_query_queue_select_ms / args.num_queries)
                  << " avg_query_frontier_sort_ms="
                  << (aggregate_query_frontier_sort_ms / args.num_queries)
                  << " avg_query_tail_merge_ms="
                  << (aggregate_query_tail_merge_ms / args.num_queries)
                  << " avg_query_candidate_sort_ms="
                  << (aggregate_query_candidate_sort_ms / args.num_queries)
                  << " avg_query_hash_rebuild_ms="
                  << (aggregate_query_hash_rebuild_ms / args.num_queries)
                  << " avg_visited="
                  << (static_cast<double>(total_visited) / args.num_queries)
                  << " avg_expanded="
                  << (static_cast<double>(total_expanded) / args.num_queries)
                  << std::endl;

        const double denom = static_cast<double>(args.num_queries * args.top_k);
        std::cout << "[topoanns_visited_overlap] top_l=" << top_l
                  << " visited_overlap@" << args.top_k
                  << "=" << (static_cast<double>(total_gt_visited) / denom)
                  << std::endl;
        for (const auto& total : search_totals) {
            std::cout << "[topoanns_overlap] top_l=" << top_l
                      << " threshold=" << total.threshold
                      << " overlap_recall@" << args.top_k
                      << "=" << (static_cast<double>(total.matched) / denom)
                      << std::endl;
            std::cout << "[topoanns_overlap_searchpq] top_l=" << top_l
                      << " threshold=" << total.threshold
                      << " overlap_recall@" << args.top_k
                      << "=" << (static_cast<double>(total.matched) / denom)
                      << std::endl;
        }
        if (resources.has_pq2_index()) {
            for (const auto& total : pq2_totals) {
                std::cout << "[topoanns_overlap_pq2] top_l=" << top_l
                          << " threshold=" << total.threshold
                          << " overlap_recall@" << args.top_k
                          << "=" << (static_cast<double>(total.matched) / denom)
                          << std::endl;
                std::cout << "[topoanns_rerank_set_overlap] top_l=" << top_l
                          << " rerank_top_n=" << total.threshold
                          << " overlap_recall@" << args.top_k
                          << "=" << (static_cast<double>(total.matched) / denom)
                          << std::endl;
            }
        }
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(ParseArgs(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_overlap_eval] " << e.what() << std::endl;
        return 1;
    }
}
