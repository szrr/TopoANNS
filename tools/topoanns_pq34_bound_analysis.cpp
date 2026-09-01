#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/rvq_entry_provider.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/vector_page_layout.hpp"
#include "topoanns/vector_store_builder.hpp"
#include "../src/search/fused_rerank_device.hpp"
#include "../src/search/topology_search_kernel.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

struct Args {
    std::filesystem::path index_dir;
    std::filesystem::path rvq_model;
    std::filesystem::path query_bin;
    std::filesystem::path pq2_pivots;
    std::filesystem::path pq2_codes;
    std::filesystem::path pq2_error_bounds;
    std::filesystem::path output_json;
    std::size_t num_queries = 0;
    std::size_t batch_size = 256;
    std::size_t top_k = 10;
    std::size_t top_l = 256;
    std::size_t top_n = 128;
    std::size_t rank_tile_size = 16;
    std::size_t search_width = 2;
    std::size_t max_expansions = 4096;
    std::size_t rvq_entry_count = 128;
};

struct FloatMatrix {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<float> values;
};

struct RunningStats {
    std::size_t count = 0;
    double min = std::numeric_limits<double>::infinity();
    double max = -std::numeric_limits<double>::infinity();
    long double sum = 0.0;

    void Add(double value) {
        ++count;
        min = std::min(min, value);
        max = std::max(max, value);
        sum += value;
    }

    double Avg() const {
        return count == 0 ? 0.0 : static_cast<double>(sum / static_cast<long double>(count));
    }
};

struct AnalysisStats {
    RunningStats error_bound_l2;
    RunningStats refine_approx_sq_distance;
    RunningStats threshold_sq_distance;
    RunningStats lower_bound_sq_distance;
    RunningStats filtered_exact_sq_distance;
    RunningStats lower_minus_threshold;
    RunningStats first_filtered_rank;
    RunningStats filtered_count_per_query;
    std::size_t filtered_total = 0;
    std::size_t queries_with_filter = 0;
    std::size_t safety_violations = 0;
    double worst_violation = 0.0;
    std::vector<std::size_t> first_filtered_rank_histogram;
    std::vector<std::size_t> filtered_count_histogram;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_pq34_bound_analysis"
        << " --index-dir <path>"
        << " --rvq-model <path>"
        << " --query-bin <path>"
        << " --pq2-pivots <path>"
        << " --pq2-codes <path>"
        << " --pq2-error-bounds <path>"
        << " --output-json <path>"
        << " --num-queries <count>"
        << " --top-l <count>"
        << " --top-n <count>"
        << " [--batch-size <count>]"
        << " [--top-k <count>]"
        << " [--rank-tile-size <count>]"
        << " [--search-width <count>]"
        << " [--max-expansions <count>]"
        << " [--rvq-entry-count <count>]"
        << std::endl;
    std::exit(1);
}

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
        throw std::runtime_error("Failed to open bin file: " + path.string());
    }
    std::int32_t rows_i32 = 0;
    std::int32_t cols_i32 = 0;
    in.read(reinterpret_cast<char*>(&rows_i32), sizeof(rows_i32));
    in.read(reinterpret_cast<char*>(&cols_i32), sizeof(cols_i32));
    if (!in.good() || rows_i32 < 0 || cols_i32 < 0) {
        throw std::runtime_error("Invalid bin header: " + path.string());
    }
    BinBlock<T> block;
    block.rows = static_cast<std::size_t>(rows_i32);
    block.cols = static_cast<std::size_t>(cols_i32);
    block.data.resize(block.rows * block.cols);
    if (!block.data.empty()) {
        in.read(reinterpret_cast<char*>(block.data.data()),
                static_cast<std::streamsize>(block.data.size() * sizeof(T)));
        if (!in.good()) {
            throw std::runtime_error("Short read in bin file: " + path.string());
        }
    }
    return block;
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
        } else if (flag == "--top-l") {
            args.top_l = std::stoull(read_value("--top-l"));
        } else if (flag == "--top-n") {
            args.top_n = std::stoull(read_value("--top-n"));
        } else if (flag == "--rank-tile-size") {
            args.rank_tile_size = std::stoull(read_value("--rank-tile-size"));
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
        args.pq2_pivots.empty() || args.pq2_codes.empty() || args.pq2_error_bounds.empty() ||
        args.output_json.empty() || args.num_queries == 0 || args.top_l == 0 || args.top_n == 0) {
        Usage();
    }
    if (args.top_n > args.top_l) {
        throw std::runtime_error("top-n must be <= top-l.");
    }
    if (args.top_k == 0 || args.top_k > 128) {
        throw std::runtime_error("top-k must be in [1, 128].");
    }
    return args;
}

double ComputeSquaredL2(const float* lhs, const float* rhs, std::size_t dim) {
    double sum = 0.0;
    for (std::size_t d = 0; d < dim; ++d) {
        const double diff = static_cast<double>(lhs[d]) - static_cast<double>(rhs[d]);
        sum += diff * diff;
    }
    return sum;
}

double ComputeSquaredNorm(const float* values, std::size_t dim) {
    double sum = 0.0;
    for (std::size_t d = 0; d < dim; ++d) {
        const double value = static_cast<double>(values[d]);
        sum += value * value;
    }
    return sum;
}

double SquaredL2LowerBound(double approx_sq_distance, double error_bound_l2) {
    const double approx_l2 = std::sqrt(std::max(approx_sq_distance, 0.0));
    const double lower_l2 = std::max(0.0, approx_l2 - error_bound_l2);
    return lower_l2 * lower_l2;
}

void ReadPageFloats(int fd,
                    std::uint64_t byte_offset,
                    std::vector<float>* out_page_floats) {
    std::uint8_t* raw = reinterpret_cast<std::uint8_t*>(out_page_floats->data());
    std::size_t remaining = out_page_floats->size() * sizeof(float);
    while (remaining > 0) {
        const ssize_t read_result =
            pread(fd, raw, remaining, static_cast<off_t>(byte_offset));
        if (read_result < 0) {
            throw std::runtime_error("pread failed while loading vector page.");
        }
        if (read_result == 0) {
            throw std::runtime_error("Unexpected EOF while loading vector page.");
        }
        raw += read_result;
        remaining -= static_cast<std::size_t>(read_result);
        byte_offset += static_cast<std::uint64_t>(read_result);
    }
}

std::string FormatStats(const RunningStats& stats) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "count=" << stats.count
        << " min=" << (stats.count == 0 ? 0.0 : stats.min)
        << " max=" << (stats.count == 0 ? 0.0 : stats.max)
        << " avg=" << stats.Avg();
    return out.str();
}

void WriteRunningStatsJson(std::ofstream& out, const RunningStats& stats) {
    out << "{";
    out << "\"count\":" << stats.count << ",";
    out << "\"min\":" << (stats.count == 0 ? 0.0 : stats.min) << ",";
    out << "\"max\":" << (stats.count == 0 ? 0.0 : stats.max) << ",";
    out << "\"avg\":" << stats.Avg();
    out << "}";
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

int Main(const Args& args) {
    const FloatMatrix queries = LoadFloatMatrix(args.query_bin);
    if (args.num_queries > queries.rows) {
        throw std::runtime_error("num-queries exceeds query rows.");
    }

    topoanns::SearchResources resources =
        topoanns::SearchResources::FromTopologyFile(args.index_dir / "topology.bin");
    resources.LoadPqIndex(args.index_dir / "_pq_pivots.bin",
                          args.index_dir / "_pq_compressed.bin");
    resources.LoadPq2Index(args.pq2_pivots, args.pq2_codes, args.pq2_error_bounds);
    topoanns::RvqModel rvq_model = topoanns::RvqModel::Load(args.rvq_model);
    rvq_model.WarmUp();

    const auto error_bounds_block = ReadBinBlock<float>(args.pq2_error_bounds);
    if (error_bounds_block.cols != 1 || error_bounds_block.rows != resources.num_nodes()) {
        throw std::runtime_error("PQ34 error-bound shape does not match node count.");
    }
    const std::vector<float>& error_bounds = error_bounds_block.data;

    const std::filesystem::path vector_store_path = args.index_dir / "vectors.ssd";
    const topoanns::VectorStoreHeader vector_store_header =
        topoanns::VectorStoreBuilder::ReadHeader(vector_store_path);
    const topoanns::VectorPageLayout layout = topoanns::VectorPageLayout::CreateFromVectorBytes(
        static_cast<std::size_t>(vector_store_header.vector_bytes),
        static_cast<std::size_t>(vector_store_header.page_size_bytes));
    if (vector_store_header.scalar_kind != static_cast<std::uint32_t>(topoanns::ScalarKind::kFloat32)) {
        throw std::runtime_error("This analysis tool currently supports only float32 vector stores.");
    }
    const std::size_t dim = vector_store_header.dim;

    const int fd = open(vector_store_path.c_str(), O_RDONLY);
    if (fd < 0) {
        throw std::runtime_error("Failed to open vectors.ssd.");
    }

    AnalysisStats stats;
    stats.first_filtered_rank_histogram.assign(args.top_n + 1, 0);
    stats.filtered_count_histogram.assign(args.top_n + 1, 0);

    try {
        for (std::size_t query_offset = 0; query_offset < args.num_queries;
             query_offset += args.batch_size) {
            const std::size_t batch_queries =
                std::min(args.batch_size, args.num_queries - query_offset);
            const float* begin =
                queries.values.data() + query_offset * static_cast<std::size_t>(queries.cols);
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
            params.top_k = args.top_l;
            params.top_l = args.top_l;
            params.candidate_queue_size = args.top_l;
            params.search_width = args.search_width;
            params.max_expansions = args.max_expansions;

            topoanns::detail::DeviceTopologyBatchResult topology_result =
                topoanns::detail::RunTopologySearchKernelBatchDevice(resources, oracle,
                                                                     entry_batch, params);
            const topoanns::detail::Pq2RefineBatchResult pq2_result =
                topoanns::detail::RunPq2RefineBatchDevice(resources, topology_result,
                                                          batch_query_buffer, batch_queries,
                                                          args.top_l);
            const std::vector<topoanns::detail::DeviceTopologyCandidate> host_pq2_candidates =
                pq2_result.candidate_buffer.CopyToHost();

            std::unordered_set<std::uint64_t> unique_page_ids;
            unique_page_ids.reserve(batch_queries * args.top_n);
            for (std::size_t query_id = 0; query_id < batch_queries; ++query_id) {
                const auto* candidates = host_pq2_candidates.data() + query_id * args.top_l;
                for (std::size_t rank = 0; rank < args.top_n; ++rank) {
                    const std::uint32_t node_id = candidates[rank].node_id;
                    if (node_id == topoanns::kInvalidNodeId) {
                        continue;
                    }
                    unique_page_ids.insert(static_cast<std::uint64_t>(node_id) /
                                           layout.vectors_per_page());
                }
            }

            std::unordered_map<std::uint64_t, std::vector<float>> page_cache;
            page_cache.reserve(unique_page_ids.size());
            for (const std::uint64_t page_id : unique_page_ids) {
                std::vector<float> page(layout.page_size_bytes() / sizeof(float), 0.0f);
                const std::uint64_t byte_offset =
                    layout.header_bytes() + page_id * layout.page_size_bytes();
                ReadPageFloats(fd, byte_offset, &page);
                page_cache.emplace(page_id, std::move(page));
            }

            std::vector<float> best_exact(args.top_k, std::numeric_limits<float>::infinity());
            for (std::size_t query_id = 0; query_id < batch_queries; ++query_id) {
                const auto* candidates = host_pq2_candidates.data() + query_id * args.top_l;
                const float* query = batch_query_buffer.data() + query_id * dim;
                const double query_norm_sq = ComputeSquaredNorm(query, dim);

                std::vector<double> exact_by_rank(args.top_n, std::numeric_limits<double>::infinity());
                for (std::size_t rank = 0; rank < args.top_n; ++rank) {
                    const std::uint32_t node_id = candidates[rank].node_id;
                    if (node_id == topoanns::kInvalidNodeId) {
                        continue;
                    }
                    const std::uint64_t page_id =
                        static_cast<std::uint64_t>(node_id) / layout.vectors_per_page();
                    const std::uint32_t slot_id =
                        static_cast<std::uint32_t>(node_id % layout.vectors_per_page());
                    const auto page_it = page_cache.find(page_id);
                    if (page_it == page_cache.end()) {
                        throw std::runtime_error("Vector page missing from cache.");
                    }
                    const float* vector =
                        page_it->second.data() + static_cast<std::size_t>(slot_id) * dim;
                    exact_by_rank[rank] = ComputeSquaredL2(query, vector, dim);
                }

                std::fill(best_exact.begin(), best_exact.end(),
                          std::numeric_limits<float>::infinity());
                double threshold = std::numeric_limits<double>::infinity();
                std::size_t filtered_count = 0;
                std::size_t first_filtered_rank = 0;

                for (std::size_t rank_begin = 0; rank_begin < args.top_n;
                     rank_begin += args.rank_tile_size) {
                    const std::size_t rank_end =
                        std::min(args.top_n, rank_begin + args.rank_tile_size);
                    for (std::size_t rank = rank_begin; rank < rank_end; ++rank) {
                        const std::uint32_t node_id = candidates[rank].node_id;
                        if (node_id == topoanns::kInvalidNodeId) {
                            continue;
                        }

                        const double approx_sq_distance =
                            std::max(0.0, static_cast<double>(candidates[rank].distance) -
                                              query_norm_sq);
                        const double error_bound_l2 =
                            static_cast<double>(error_bounds[node_id]);
                        const double lower_bound_sq =
                            SquaredL2LowerBound(approx_sq_distance, error_bound_l2);

                        bool keep = true;
                        if (std::isfinite(threshold) && lower_bound_sq > threshold) {
                            keep = false;
                            ++filtered_count;
                            ++stats.filtered_total;
                            stats.error_bound_l2.Add(error_bound_l2);
                            stats.refine_approx_sq_distance.Add(approx_sq_distance);
                            stats.threshold_sq_distance.Add(threshold);
                            stats.lower_bound_sq_distance.Add(lower_bound_sq);
                            stats.filtered_exact_sq_distance.Add(exact_by_rank[rank]);
                            stats.lower_minus_threshold.Add(lower_bound_sq - threshold);
                            if (first_filtered_rank == 0) {
                                first_filtered_rank = rank + 1;
                            }
                            if (exact_by_rank[rank] < threshold) {
                                ++stats.safety_violations;
                                stats.worst_violation = std::max(
                                    stats.worst_violation, threshold - exact_by_rank[rank]);
                            }
                        }

                        if (!keep) {
                            continue;
                        }

                        const float exact = static_cast<float>(exact_by_rank[rank]);
                        if (!(exact < best_exact.back())) {
                            continue;
                        }
                        std::size_t insert = args.top_k - 1;
                        while (insert > 0 && exact < best_exact[insert - 1]) {
                            best_exact[insert] = best_exact[insert - 1];
                            --insert;
                        }
                        best_exact[insert] = exact;
                    }
                    threshold = static_cast<double>(best_exact.back());
                }

                stats.filtered_count_per_query.Add(static_cast<double>(filtered_count));
                ++stats.filtered_count_histogram[std::min(filtered_count, args.top_n)];
                if (first_filtered_rank == 0) {
                    ++stats.first_filtered_rank_histogram[0];
                } else {
                    ++stats.queries_with_filter;
                    stats.first_filtered_rank.Add(static_cast<double>(first_filtered_rank));
                    ++stats.first_filtered_rank_histogram[first_filtered_rank];
                }
            }

            std::cout << "[topoanns_pq34_bound_analysis_progress] top_l=" << args.top_l
                      << " top_n=" << args.top_n
                      << " finished queries " << (query_offset + batch_queries)
                      << " / " << args.num_queries << std::endl;
        }
    } catch (...) {
        close(fd);
        throw;
    }
    close(fd);

    std::cout << "[topoanns_pq34_bound_analysis] filtered_total=" << stats.filtered_total
              << " queries_with_filter=" << stats.queries_with_filter
              << " safety_violations=" << stats.safety_violations
              << " worst_violation=" << stats.worst_violation << std::endl;
    std::cout << "[topoanns_pq34_bound_analysis] error_bound_l2 "
              << FormatStats(stats.error_bound_l2) << std::endl;
    std::cout << "[topoanns_pq34_bound_analysis] refine_approx_sq_distance "
              << FormatStats(stats.refine_approx_sq_distance) << std::endl;
    std::cout << "[topoanns_pq34_bound_analysis] threshold_sq_distance "
              << FormatStats(stats.threshold_sq_distance) << std::endl;
    std::cout << "[topoanns_pq34_bound_analysis] lower_bound_sq_distance "
              << FormatStats(stats.lower_bound_sq_distance) << std::endl;
    std::cout << "[topoanns_pq34_bound_analysis] filtered_exact_sq_distance "
              << FormatStats(stats.filtered_exact_sq_distance) << std::endl;
    std::cout << "[topoanns_pq34_bound_analysis] lower_minus_threshold "
              << FormatStats(stats.lower_minus_threshold) << std::endl;
    std::cout << "[topoanns_pq34_bound_analysis] first_filtered_rank "
              << FormatStats(stats.first_filtered_rank) << std::endl;
    std::cout << "[topoanns_pq34_bound_analysis] filtered_count_per_query "
              << FormatStats(stats.filtered_count_per_query) << std::endl;

    std::ofstream out(args.output_json, std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output json: " + args.output_json.string());
    }
    out << "{";
    out << "\"num_queries\":" << args.num_queries << ",";
    out << "\"top_l\":" << args.top_l << ",";
    out << "\"top_n\":" << args.top_n << ",";
    out << "\"top_k\":" << args.top_k << ",";
    out << "\"rank_tile_size\":" << args.rank_tile_size << ",";
    out << "\"filtered_total\":" << stats.filtered_total << ",";
    out << "\"queries_with_filter\":" << stats.queries_with_filter << ",";
    out << "\"safety_violations\":" << stats.safety_violations << ",";
    out << "\"worst_violation\":" << stats.worst_violation << ",";
    out << "\"error_bound_l2\":";
    WriteRunningStatsJson(out, stats.error_bound_l2);
    out << ",\"refine_approx_sq_distance\":";
    WriteRunningStatsJson(out, stats.refine_approx_sq_distance);
    out << ",\"threshold_sq_distance\":";
    WriteRunningStatsJson(out, stats.threshold_sq_distance);
    out << ",\"lower_bound_sq_distance\":";
    WriteRunningStatsJson(out, stats.lower_bound_sq_distance);
    out << ",\"filtered_exact_sq_distance\":";
    WriteRunningStatsJson(out, stats.filtered_exact_sq_distance);
    out << ",\"lower_minus_threshold\":";
    WriteRunningStatsJson(out, stats.lower_minus_threshold);
    out << ",\"first_filtered_rank\":";
    WriteRunningStatsJson(out, stats.first_filtered_rank);
    out << ",\"filtered_count_per_query\":";
    WriteRunningStatsJson(out, stats.filtered_count_per_query);
    out << ",\"first_filtered_rank_histogram\":";
    WriteHistogramJson(out, stats.first_filtered_rank_histogram);
    out << ",\"filtered_count_histogram\":";
    WriteHistogramJson(out, stats.filtered_count_histogram);
    out << "}";
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(ParseArgs(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_pq34_bound_analysis] " << e.what() << std::endl;
        return 1;
    }
}
