#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/rvq_entry_provider.hpp"
#include "topoanns/search_resources.hpp"
#include "../src/search/fused_rerank_device.hpp"
#include "../src/search/topology_search_kernel.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
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
    std::filesystem::path pq2_error_bounds;
    std::filesystem::path output_json;
    std::size_t num_queries = 0;
    std::size_t batch_size = 256;
    std::size_t top_k = 10;
    std::vector<std::size_t> top_l_values;
    std::size_t search_width = 2;
    std::size_t max_expansions = 4096;
    std::size_t rvq_entry_count = 128;
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

struct RankStats {
    std::size_t total = 0;
    std::size_t found = 0;
    std::size_t not_found = 0;
    std::size_t min_rank = std::numeric_limits<std::size_t>::max();
    std::size_t max_rank = 0;
    double rank_sum = 0.0;
    std::vector<std::size_t> histogram;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_pq2_rank_distribution"
        << " --index-dir <path>"
        << " --rvq-model <path>"
        << " --query-bin <path>"
        << " --gt-bin <path>"
        << " --pq2-pivots <path>"
        << " --pq2-codes <path>"
        << " [--pq2-error-bounds <path>]"
        << " --output-json <path>"
        << " --num-queries <count>"
        << " --top-l-values <csv>"
        << " [--batch-size <count>]"
        << " [--top-k <count>]"
        << " [--search-width <count>]"
        << " [--max-expansions <count>]"
        << " [--rvq-entry-count <count>]"
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
        } else if (flag == "--pq2-error-bounds") {
            args.pq2_error_bounds = read_value("--pq2-error-bounds");
        } else if (flag == "--output-json") {
            args.output_json = read_value("--output-json");
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
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }

    if (args.index_dir.empty() || args.rvq_model.empty() || args.query_bin.empty() ||
        args.gt_bin.empty() || args.pq2_pivots.empty() || args.pq2_codes.empty() ||
        args.output_json.empty() || args.num_queries == 0 || args.top_l_values.empty()) {
        Usage();
    }
    std::sort(args.top_l_values.begin(), args.top_l_values.end());
    args.top_l_values.erase(
        std::unique(args.top_l_values.begin(), args.top_l_values.end()),
        args.top_l_values.end());
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

std::size_t FindRank(const topoanns::detail::DeviceTopologyCandidate* candidates,
                     std::size_t candidate_limit,
                     std::uint32_t target) {
    for (std::size_t rank = 0; rank < candidate_limit; ++rank) {
        if (candidates[rank].node_id == topoanns::kInvalidNodeId) {
            break;
        }
        if (candidates[rank].node_id == target) {
            return rank + 1;
        }
    }
    return 0;
}

void UpdateRankStats(RankStats* stats, std::size_t rank) {
    ++stats->total;
    if (rank == 0) {
        ++stats->not_found;
        ++stats->histogram[0];
        return;
    }
    ++stats->found;
    stats->min_rank = std::min(stats->min_rank, rank);
    stats->max_rank = std::max(stats->max_rank, rank);
    stats->rank_sum += static_cast<double>(rank);
    ++stats->histogram[rank];
}

void WriteHistogramJson(std::ofstream& out, const std::vector<std::size_t>& histogram) {
    out << "[";
    for (std::size_t i = 0; i < histogram.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << histogram[i];
    }
    out << "]";
}

void WriteStatsJson(std::ofstream& out, const char* name, const RankStats& stats) {
    out << "\"" << name << "\":{";
    out << "\"total\":" << stats.total << ",";
    out << "\"found\":" << stats.found << ",";
    out << "\"not_found\":" << stats.not_found << ",";
    out << "\"min_rank\":" << (stats.found == 0 ? 0 : stats.min_rank) << ",";
    out << "\"max_rank\":" << stats.max_rank << ",";
    out << "\"avg_rank\":" << (stats.found == 0 ? 0.0 : stats.rank_sum / stats.found) << ",";
    out << "\"histogram\":";
    WriteHistogramJson(out, stats.histogram);
    out << "}";
}

int Main(const Args& args) {
    const FloatMatrix queries = LoadFloatMatrix(args.query_bin);
    const IntMatrix gt = LoadIntMatrix(args.gt_bin);
    if (args.num_queries > queries.rows || args.num_queries > gt.rows) {
        throw std::runtime_error("num-queries exceeds query/gt rows.");
    }
    if (gt.cols < args.top_k) {
        throw std::runtime_error("gt cols must be >= top-k.");
    }

    topoanns::SearchResources resources =
        topoanns::SearchResources::FromTopologyFile(args.index_dir / "topology.bin");
    resources.LoadPqIndex(args.index_dir / "_pq_pivots.bin",
                          args.index_dir / "_pq_compressed.bin");
    resources.LoadPq2Index(args.pq2_pivots, args.pq2_codes, args.pq2_error_bounds);
    topoanns::RvqModel rvq_model = topoanns::RvqModel::Load(args.rvq_model);
    rvq_model.WarmUp();

    std::ofstream json_out(args.output_json, std::ios::trunc);
    if (!json_out.is_open()) {
        throw std::runtime_error("Failed to open output json: " + args.output_json.string());
    }

    json_out << "{";
    json_out << "\"num_queries\":" << args.num_queries << ",";
    json_out << "\"top_k\":" << args.top_k << ",";
    json_out << "\"top_l_results\":[";

    for (std::size_t top_l_index = 0; top_l_index < args.top_l_values.size(); ++top_l_index) {
        const std::size_t top_l = args.top_l_values[top_l_index];
        RankStats topology_stats;
        RankStats pq2_stats;
        topology_stats.histogram.assign(top_l + 1, 0);
        pq2_stats.histogram.assign(top_l + 1, 0);
        double pq2_total_ms = 0.0;
        double pq2_query_upload_ms = 0.0;
        double pq2_zero_fill_ms = 0.0;
        double pq2_table_kernel_ms = 0.0;
        double pq2_table_download_ms = 0.0;
        double pq2_refine_kernel_ms = 0.0;

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
                                                         args.rvq_entry_count, 0, &entry_profile);

            const topoanns::PqDistanceOracle oracle =
                topoanns::PqDistanceOracle::FromFloatQueries(resources, batch_query_buffer,
                                                             batch_queries);
            topoanns::TopologySearchParams params;
            params.top_k = top_l;
            params.top_l = top_l;
            params.candidate_queue_size = top_l;
            params.search_width = args.search_width;
            params.max_expansions = args.max_expansions;

            topoanns::detail::DeviceTopologyBatchResult topology_result =
                topoanns::detail::RunTopologySearchKernelBatchDevice(resources, oracle,
                                                                     entry_batch, params);
            const std::vector<topoanns::detail::DeviceTopologyCandidate> host_topology_candidates =
                topology_result.candidate_buffer.CopyToHost();

            const topoanns::detail::Pq2RefineBatchResult pq2_result =
                topoanns::detail::RunPq2RefineBatchDevice(resources, topology_result,
                                                          batch_query_buffer, batch_queries,
                                                          top_l);
            pq2_total_ms += pq2_result.pq_profile.total_ms;
            pq2_query_upload_ms += pq2_result.pq_profile.query_upload_ms;
            pq2_zero_fill_ms += pq2_result.pq_profile.zero_fill_ms;
            pq2_table_kernel_ms += pq2_result.pq_profile.kernel_ms;
            pq2_table_download_ms += pq2_result.pq_profile.table_download_ms;
            pq2_refine_kernel_ms += pq2_result.kernel_ms;
            const std::vector<topoanns::detail::DeviceTopologyCandidate> host_pq2_candidates =
                pq2_result.candidate_buffer.CopyToHost();

            for (std::size_t batch_query_id = 0; batch_query_id < batch_queries; ++batch_query_id) {
                const auto* topology_candidates =
                    host_topology_candidates.data() + batch_query_id * top_l;
                const auto* pq2_candidates =
                    host_pq2_candidates.data() + batch_query_id * top_l;
                const std::int32_t* gt_row =
                    gt.values.data() +
                    (query_offset + batch_query_id) * static_cast<std::size_t>(gt.cols);
                for (std::size_t gt_rank = 0; gt_rank < args.top_k; ++gt_rank) {
                    const std::uint32_t target = static_cast<std::uint32_t>(gt_row[gt_rank]);
                    UpdateRankStats(&topology_stats,
                                    FindRank(topology_candidates, top_l, target));
                    UpdateRankStats(&pq2_stats, FindRank(pq2_candidates, top_l, target));
                }
            }

            std::cout << "[topoanns_pq2_rank_progress] top_l=" << top_l
                      << " finished queries " << (query_offset + batch_queries)
                      << " / " << args.num_queries << std::endl;
        }

        std::cout << "[topoanns_pq2_rank_stats] top_l=" << top_l
                  << " stage=topology"
                  << " found=" << topology_stats.found
                  << " not_found=" << topology_stats.not_found
                  << " min_rank=" << (topology_stats.found == 0 ? 0 : topology_stats.min_rank)
                  << " max_rank=" << topology_stats.max_rank
                  << " avg_rank="
                  << (topology_stats.found == 0 ? 0.0
                                               : topology_stats.rank_sum / topology_stats.found)
                  << std::endl;
        std::cout << "[topoanns_pq2_rank_stats] top_l=" << top_l
                  << " stage=pq2"
                  << " found=" << pq2_stats.found
                  << " not_found=" << pq2_stats.not_found
                  << " min_rank=" << (pq2_stats.found == 0 ? 0 : pq2_stats.min_rank)
                  << " max_rank=" << pq2_stats.max_rank
                  << " avg_rank="
                  << (pq2_stats.found == 0 ? 0.0 : pq2_stats.rank_sum / pq2_stats.found)
                  << std::endl;
        std::cout << "[topoanns_pq2_rank_profile] top_l=" << top_l
                  << " pq2_total_ms=" << pq2_total_ms
                  << " pq2_query_upload_ms=" << pq2_query_upload_ms
                  << " pq2_zero_fill_ms=" << pq2_zero_fill_ms
                  << " pq2_table_kernel_ms=" << pq2_table_kernel_ms
                  << " pq2_table_download_ms=" << pq2_table_download_ms
                  << " pq2_refine_kernel_ms=" << pq2_refine_kernel_ms
                  << std::endl;

        if (top_l_index != 0) {
            json_out << ",";
        }
        json_out << "{";
        json_out << "\"top_l\":" << top_l << ",";
        WriteStatsJson(json_out, "topology", topology_stats);
        json_out << ",";
        WriteStatsJson(json_out, "pq2", pq2_stats);
        json_out << "}";
    }

    json_out << "]}";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(ParseArgs(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_pq2_rank_distribution] " << e.what() << std::endl;
        return 1;
    }
}
