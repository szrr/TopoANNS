#include "topoanns/bvecs_io.hpp"
#include "topoanns/pq_distance_oracle.hpp"
#include "topoanns/rvq_entry_provider.hpp"
#include "topoanns/search_resources.hpp"
#include "topoanns/vector_store_builder.hpp"
#include "../src/search/fused_rerank_device.hpp"
#include "../src/search/topology_search_kernel.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

namespace {

constexpr std::size_t kRerankLearnedStopHeadProbeCount = 8;

struct Args {
    std::filesystem::path index_dir;
    std::filesystem::path rvq_model;
    std::filesystem::path learn_bvecs;
    std::filesystem::path query_bin;
    std::filesystem::path gt_bin;
    std::filesystem::path output_csv;
    std::filesystem::path pq2_pivots;
    std::filesystem::path pq2_codes;
    std::filesystem::path pq2_error_bounds;
    std::size_t query_start = 0;
    std::size_t num_queries = 0;
    std::size_t batch_size = 256;
    std::size_t top_k = 10;
    std::vector<std::size_t> top_l_values;
    std::size_t search_width = 2;
    std::size_t max_expansions = 4096;
    std::size_t rvq_entry_count = 128;
    std::uint32_t cuda_device = 0;
};

struct OracleCandidate {
    float distance = std::numeric_limits<float>::infinity();
    std::uint32_t node_id = topoanns::kInvalidNodeId;
};

struct RunningTopK {
    std::vector<OracleCandidate> values;
};

class HostVectorStoreReader {
public:
    HostVectorStoreReader(const std::filesystem::path& path,
                          const topoanns::VectorPageLayout& layout)
        : path_(path), layout_(layout) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to open vector store: " + path.string());
        }
    }

    ~HostVectorStoreReader() {
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    const std::uint8_t* ReadVector(std::uint64_t node_id) {
        const topoanns::VectorPageAddress address = layout_.Resolve(node_id);
        auto it = pages_.find(address.page_id);
        if (it == pages_.end()) {
            LoadPage(address.page_id);
            it = pages_.find(address.page_id);
        }
        return it->second.data() + static_cast<std::size_t>(address.slot_id) * layout_.vector_bytes();
    }

private:
    void LoadPage(std::uint64_t page_id) {
        std::vector<std::uint8_t> page(layout_.page_size_bytes(), 0);
        const off_t offset = static_cast<off_t>(
            layout_.header_bytes() + page_id * layout_.page_size_bytes());
        const ssize_t bytes_read =
            ::pread(fd_, page.data(), page.size(), offset);
        if (bytes_read != static_cast<ssize_t>(page.size())) {
            throw std::runtime_error("Failed to read vector page from " + path_.string());
        }
        pages_[page_id] = std::move(page);
        page_order_.push_back(page_id);
        constexpr std::size_t kMaxCachedPages = 4096;
        if (page_order_.size() > kMaxCachedPages) {
            pages_.erase(page_order_.front());
            page_order_.pop_front();
        }
    }

    std::filesystem::path path_;
    topoanns::VectorPageLayout layout_;
    int fd_ = -1;
    std::unordered_map<std::uint64_t, std::vector<std::uint8_t>> pages_;
    std::deque<std::uint64_t> page_order_;
};

[[noreturn]] void Usage() {
    std::cerr << "Usage: topoanns_learned_rerank_stop_dataset"
              << " --index-dir <path>"
              << " --rvq-model <path>"
              << " (--learn-bvecs <path> | --query-bin <path> --gt-bin <path>)"
              << " --output-csv <path>"
              << " --pq2-pivots <path>"
              << " --pq2-codes <path>"
              << " --pq2-error-bounds <path>"
              << " --num-queries <count>"
              << " --top-l-values <csv>"
              << " [--query-start <count>]"
              << " [--batch-size <count>]"
              << " [--top-k <count>]"
              << " [--search-width <count>]"
              << " [--max-expansions <count>]"
              << " [--rvq-entry-count <count>]"
              << " [--cuda-device <id>]"
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
    return values;
}

void RequireL40Device(std::uint32_t device_id, const char* context) {
    cudaDeviceProp props{};
    const cudaError_t status = cudaGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) + ": cudaGetDeviceProperties failed.");
    }
    const std::string name(props.name);
    if (name.find("L40") == std::string::npos) {
        throw std::runtime_error(std::string(context) + ": expected L40 but got " + name);
    }
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
        } else if (flag == "--learn-bvecs") {
            args.learn_bvecs = read_value("--learn-bvecs");
        } else if (flag == "--query-bin") {
            args.query_bin = read_value("--query-bin");
        } else if (flag == "--gt-bin") {
            args.gt_bin = read_value("--gt-bin");
        } else if (flag == "--output-csv") {
            args.output_csv = read_value("--output-csv");
        } else if (flag == "--pq2-pivots") {
            args.pq2_pivots = read_value("--pq2-pivots");
        } else if (flag == "--pq2-codes") {
            args.pq2_codes = read_value("--pq2-codes");
        } else if (flag == "--pq2-error-bounds") {
            args.pq2_error_bounds = read_value("--pq2-error-bounds");
        } else if (flag == "--query-start") {
            args.query_start = std::stoull(read_value("--query-start"));
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
        } else if (flag == "--cuda-device") {
            args.cuda_device = static_cast<std::uint32_t>(std::stoul(read_value("--cuda-device")));
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }

    const bool has_learn_bvecs = !args.learn_bvecs.empty();
    const bool has_query_gt = !args.query_bin.empty() && !args.gt_bin.empty();
    if (args.index_dir.empty() || args.rvq_model.empty() ||
        args.output_csv.empty() || args.pq2_pivots.empty() || args.pq2_codes.empty() ||
        args.pq2_error_bounds.empty() || args.num_queries == 0 || args.top_l_values.empty()) {
        Usage();
    }
    if (!has_learn_bvecs && !has_query_gt) {
        Usage();
    }
    if (!args.query_bin.empty() != !args.gt_bin.empty()) {
        Usage();
    }
    return args;
}

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
    matrix.values.resize(static_cast<std::size_t>(matrix.rows) * matrix.cols);
    in.read(reinterpret_cast<char*>(matrix.values.data()),
            static_cast<std::streamsize>(matrix.values.size() * sizeof(std::int32_t)));
    if (in.gcount() != static_cast<std::streamsize>(matrix.values.size() * sizeof(std::int32_t))) {
        throw std::runtime_error("Short read in int matrix: " + path.string());
    }
    return matrix;
}

std::vector<std::uint32_t> BuildRerankCheckpointLadder(std::size_t top_l) {
    std::vector<std::uint32_t> ladder;
    for (std::size_t prefix = 10; prefix <= top_l;) {
        ladder.push_back(static_cast<std::uint32_t>(prefix));
        std::size_t step = 2;
        if (prefix >= 1024) {
            step = 128;
        } else if (prefix >= 512) {
            step = 64;
        } else if (prefix >= 256) {
            step = 32;
        } else if (prefix >= 128) {
            step = 16;
        } else if (prefix >= 64) {
            step = 8;
        } else if (prefix >= 32) {
            step = 4;
        }
        prefix += step;
    }
    return ladder;
}

float SquaredL2LowerBound(float approx_sq_distance, float error_bound_l2) {
    const float approx_l2 = std::sqrt(std::max(approx_sq_distance, 0.0f));
    const float lower_l2 = std::max(0.0f, approx_l2 - error_bound_l2);
    return lower_l2 * lower_l2;
}

bool CandidateLess(const OracleCandidate& lhs, const OracleCandidate& rhs) {
    if (lhs.distance != rhs.distance) {
        return lhs.distance < rhs.distance;
    }
    return lhs.node_id < rhs.node_id;
}

bool InsertIntoTopK(const OracleCandidate& candidate, RunningTopK* topk, std::size_t top_k) {
    if (top_k == 0) {
        return false;
    }
    if (topk->values.size() < top_k) {
        topk->values.push_back(candidate);
        std::sort(topk->values.begin(), topk->values.end(), CandidateLess);
        return true;
    }
    if (!CandidateLess(candidate, topk->values.back())) {
        return false;
    }
    topk->values.back() = candidate;
    std::size_t insert = topk->values.size() - 1;
    while (insert > 0 && CandidateLess(topk->values[insert], topk->values[insert - 1])) {
        std::swap(topk->values[insert], topk->values[insert - 1]);
        --insert;
    }
    return true;
}

float ComputeSquaredL2(const float* query, const std::uint8_t* vector_bytes, std::size_t dim) {
    const float* vector = reinterpret_cast<const float*>(vector_bytes);
    float distance = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
        const float diff = vector[d] - query[d];
        distance += diff * diff;
    }
    return distance;
}

std::size_t ComputeTopKOverlap(const RunningTopK& current,
                               const std::vector<std::uint32_t>& oracle_ids) {
    std::size_t overlap = 0;
    for (const auto& candidate : current.values) {
        if (std::find(oracle_ids.begin(), oracle_ids.end(), candidate.node_id) != oracle_ids.end()) {
            ++overlap;
        }
    }
    return overlap;
}

std::size_t ComputeTopKOverlapWithGt(const RunningTopK& current,
                                     const std::int32_t* gt_row,
                                     std::size_t top_k) {
    std::size_t overlap = 0;
    for (const auto& candidate : current.values) {
        for (std::size_t rank = 0; rank < top_k; ++rank) {
            if (static_cast<std::int32_t>(candidate.node_id) == gt_row[rank]) {
                ++overlap;
                break;
            }
        }
    }
    return overlap;
}

std::size_t ComputeOracleGtOverlap(const std::vector<std::uint32_t>& oracle_ids,
                                   const std::int32_t* gt_row,
                                   std::size_t top_k) {
    std::size_t overlap = 0;
    for (const std::uint32_t node_id : oracle_ids) {
        for (std::size_t rank = 0; rank < top_k; ++rank) {
            if (static_cast<std::int32_t>(node_id) == gt_row[rank]) {
                ++overlap;
                break;
            }
        }
    }
    return overlap;
}

std::vector<std::uint32_t> BuildOracleTopK(const std::vector<OracleCandidate>& all_exact,
                                           std::size_t top_k) {
    std::vector<OracleCandidate> sorted = all_exact;
    std::sort(sorted.begin(), sorted.end(), CandidateLess);
    std::vector<std::uint32_t> ids;
    ids.reserve(top_k);
    for (const auto& candidate : sorted) {
        if (candidate.node_id == topoanns::kInvalidNodeId) {
            continue;
        }
        ids.push_back(candidate.node_id);
        if (ids.size() == top_k) {
            break;
        }
    }
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        if (cudaSetDevice(static_cast<int>(args.cuda_device)) != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed.");
        }
        RequireL40Device(args.cuda_device, "topoanns_learned_rerank_stop_dataset");

        std::vector<float> queries;
        std::uint32_t query_dim = 0;
        IntMatrix gt_matrix;
        const bool use_query_gt = !args.query_bin.empty();
        if (use_query_gt) {
            const FloatMatrix query_matrix = LoadFloatMatrix(args.query_bin);
            gt_matrix = LoadIntMatrix(args.gt_bin);
            if (query_matrix.rows != gt_matrix.rows) {
                throw std::runtime_error("Query rows and GT rows do not match.");
            }
            if (args.query_start + args.num_queries > query_matrix.rows) {
                throw std::runtime_error("Requested query range exceeds query.bin rows.");
            }
            query_dim = query_matrix.cols;
            const std::size_t begin =
                args.query_start * static_cast<std::size_t>(query_matrix.cols);
            const std::size_t count =
                args.num_queries * static_cast<std::size_t>(query_matrix.cols);
            queries.assign(query_matrix.values.begin() + static_cast<std::ptrdiff_t>(begin),
                           query_matrix.values.begin() +
                               static_cast<std::ptrdiff_t>(begin + count));
        } else {
            const topoanns::BvecsMetadata metadata =
                topoanns::ReadBvecsMetadata(args.learn_bvecs);
            if (args.query_start + args.num_queries > metadata.num_vectors) {
                throw std::runtime_error("Requested query range exceeds learn.bvecs rows.");
            }
            queries = topoanns::ReadBvecsRangeAsFloat32(
                args.learn_bvecs, args.query_start, args.num_queries);
            query_dim = metadata.dim;
        }

        topoanns::SearchResources resources =
            topoanns::SearchResources::FromTopologyFile(args.index_dir / "topology.bin");
        resources.LoadPqIndex(args.index_dir / "_pq_pivots.bin",
                              args.index_dir / "_pq_compressed.bin");
        resources.LoadPq2Index(args.pq2_pivots, args.pq2_codes, args.pq2_error_bounds);
        resources.LoadVectorStore(args.index_dir / "vectors.ssd");
        topoanns::RvqModel rvq_model = topoanns::RvqModel::Load(args.rvq_model);
        if (rvq_model.dim() != query_dim || resources.vector_store_header().dim != query_dim) {
            throw std::runtime_error("Query dim, RVQ dim, and vector dim must match.");
        }

        HostVectorStoreReader vector_reader(resources.vector_store_path(),
                                            resources.vector_store_layout());
        const std::size_t dim = resources.vector_store_header().dim;

        std::ofstream out(args.output_csv);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open output csv: " + args.output_csv.string());
        }
        out << "query_id,top_l,stage_index,prefix,label,"
               "budget_log2,prefix_frac,current_kth,delta_kth_last,delta_kth_prev,"
               "topk_changed_flag,entered_topk_ratio,current_topk_spread,next_window_live_ratio,"
               "next_window_best_lb_gap,current_window_exact_ratio,"
               "current_window_filtered_ratio,cumulative_exact_frac,"
               "exact_count,filtered_count,next_window_live_count,next_window_size,"
               "oracle_overlap,stop_hits_gt,full_hits_gt,stop_recall,full_recall,delta_recall\n";

        const std::vector<float> error_bounds =
            resources.pq2_error_bounds_fp32().CopyToHost();

        for (const std::size_t top_l : args.top_l_values) {
            if (top_l < args.top_k || top_l < 10U) {
                continue;
            }
            const std::vector<std::uint32_t> ladder = BuildRerankCheckpointLadder(top_l);
            topoanns::TopologySearchParams topology_params;
            topology_params.top_k = top_l;
            topology_params.top_l = top_l;
            topology_params.search_width = args.search_width;
            topology_params.max_expansions = args.max_expansions;
            topology_params.candidate_queue_size = top_l;

            for (std::size_t query_offset = 0; query_offset < args.num_queries;
                 query_offset += args.batch_size) {
                const std::size_t batch_queries =
                    std::min(args.batch_size, args.num_queries - query_offset);
                const float* begin = queries.data() + query_offset * dim;
                const float* end = begin + batch_queries * dim;
                std::vector<float> batch_query_buffer(begin, end);
                topoanns::RvqEntryProfile entry_profile;
                topoanns::DeviceEntryBatch entry_batch = rvq_model.ComputeFloat32DeviceEntryBatch(
                    batch_query_buffer, batch_queries, args.rvq_entry_count, 0, &entry_profile);
                (void)entry_profile;
                const topoanns::PqDistanceOracle oracle =
                    topoanns::PqDistanceOracle::FromFloatQueries(
                        resources, batch_query_buffer, batch_queries);
                topoanns::detail::DeviceTopologyBatchResult topology_device =
                    topoanns::detail::RunTopologySearchKernelBatchDevice(
                        resources, oracle, entry_batch, topology_params);
                topoanns::detail::Pq2RefineBatchResult refine_result =
                    topoanns::detail::RunPq2RefineBatchDevice(
                        resources, topology_device, batch_query_buffer, batch_queries, top_l);
                const std::vector<topoanns::detail::DeviceTopologyCandidate> host_candidates =
                    refine_result.candidate_buffer.CopyToHost();

                for (std::size_t local_query = 0; local_query < batch_queries; ++local_query) {
                    const std::size_t global_query = args.query_start + query_offset + local_query;
                    const float* query = batch_query_buffer.data() + local_query * dim;
                    double query_norm_square = 0.0;
                    for (std::size_t d = 0; d < dim; ++d) {
                        query_norm_square += static_cast<double>(query[d]) * query[d];
                    }
                    const auto* candidates =
                        host_candidates.data() + local_query * refine_result.candidate_capacity;

                    std::vector<OracleCandidate> oracle_exact;
                    oracle_exact.reserve(top_l);
                    for (std::size_t rank = 0; rank < top_l; ++rank) {
                        const auto candidate = candidates[rank];
                        if (!candidate.valid()) {
                            continue;
                        }
                        const std::uint32_t raw_node_id = candidate.raw_node_id();
                        const std::uint8_t* vector_bytes = vector_reader.ReadVector(raw_node_id);
                        oracle_exact.push_back(
                            {ComputeSquaredL2(query, vector_bytes, dim), raw_node_id});
                    }
                    if (oracle_exact.size() < args.top_k) {
                        continue;
                    }
                    const std::vector<std::uint32_t> oracle_topk_ids =
                        BuildOracleTopK(oracle_exact, args.top_k);
                    const std::int32_t* gt_row = nullptr;
                    std::size_t full_hits_gt = 0;
                    float full_recall = 0.0f;
                    if (use_query_gt) {
                        gt_row = gt_matrix.values.data() +
                                 global_query * static_cast<std::size_t>(gt_matrix.cols);
                        full_hits_gt =
                            ComputeOracleGtOverlap(oracle_topk_ids, gt_row, args.top_k);
                        full_recall = static_cast<float>(full_hits_gt) /
                                      static_cast<float>(args.top_k);
                    }

                    RunningTopK current_topk;
                    current_topk.values.reserve(args.top_k);
                    float previous_checkpoint_kth = std::numeric_limits<float>::infinity();
                    float previous_delta_kth = 0.0f;
                    std::size_t processed_prefix = 0;
                    std::size_t exact_count = 0;
                    std::size_t filtered_count = 0;
                    std::vector<float> candidate_lower_bounds(top_l,
                                                              std::numeric_limits<float>::infinity());
                    for (std::size_t rank = 0; rank < top_l; ++rank) {
                        const auto candidate = candidates[rank];
                        if (!candidate.valid()) {
                            continue;
                        }
                        const std::uint32_t raw_node_id = candidate.raw_node_id();
                        candidate_lower_bounds[rank] = SquaredL2LowerBound(
                            candidate.distance - static_cast<float>(query_norm_square),
                            error_bounds[raw_node_id]);
                    }

                    for (std::size_t stage_index = 0; stage_index < ladder.size(); ++stage_index) {
                        const std::size_t prefix = ladder[stage_index];
                        const std::size_t next_prefix =
                            stage_index + 1 < ladder.size() ? ladder[stage_index + 1] : prefix;
                        const std::size_t window_begin = processed_prefix;
                        const std::size_t window_end = prefix;
                        const std::size_t window_size = window_end - window_begin;
                        std::size_t window_exact = 0;
                        std::size_t window_filtered = 0;
                        std::size_t entered_topk_count = 0;

                        for (std::size_t rank = window_begin; rank < window_end; ++rank) {
                            const auto candidate = candidates[rank];
                            if (!candidate.valid()) {
                                continue;
                            }
                            const std::uint32_t raw_node_id = candidate.raw_node_id();
                            const bool threshold_ready =
                                current_topk.values.size() >= args.top_k &&
                                std::isfinite(current_topk.values[args.top_k - 1].distance);
                            if (threshold_ready) {
                                const float lower_bound = candidate_lower_bounds[rank];
                                if (lower_bound > current_topk.values[args.top_k - 1].distance) {
                                    ++filtered_count;
                                    ++window_filtered;
                                    continue;
                                }
                            }

                            const std::uint8_t* vector_bytes =
                                vector_reader.ReadVector(raw_node_id);
                            if (InsertIntoTopK(
                                    {ComputeSquaredL2(query, vector_bytes, dim), raw_node_id},
                                    &current_topk, args.top_k)) {
                                ++entered_topk_count;
                            }
                            ++exact_count;
                            ++window_exact;
                        }

                        processed_prefix = prefix;
                        if (current_topk.values.size() < args.top_k ||
                            !std::isfinite(current_topk.values[args.top_k - 1].distance)) {
                            continue;
                        }

                        const float current_kth = current_topk.values[args.top_k - 1].distance;
                        const float delta_kth_last = std::isfinite(previous_checkpoint_kth)
                                                         ? (previous_checkpoint_kth - current_kth)
                                                         : 0.0f;
                        const float topk_changed_flag = entered_topk_count == 0U ? 0.0f : 1.0f;
                        const float entered_topk_ratio =
                            static_cast<float>(entered_topk_count) /
                            static_cast<float>(args.top_k);
                        const float current_topk_spread =
                            current_kth - current_topk.values.front().distance;

                        std::size_t next_window_live_count = 0;
                        float best_live_lb = std::numeric_limits<float>::infinity();
                        const std::size_t next_probe_end =
                            std::min(next_prefix, prefix + kRerankLearnedStopHeadProbeCount);
                        for (std::size_t rank = prefix; rank < next_probe_end; ++rank) {
                            const auto candidate = candidates[rank];
                            if (!candidate.valid()) {
                                continue;
                            }
                            const float lower_bound = candidate_lower_bounds[rank];
                            if (lower_bound <= current_kth) {
                                ++next_window_live_count;
                                best_live_lb = std::min(best_live_lb, lower_bound);
                            }
                        }

                        std::size_t stop_hits_gt = 0;
                        float stop_recall = 0.0f;
                        float delta_recall = 0.0f;
                        if (gt_row != nullptr) {
                            stop_hits_gt =
                                ComputeTopKOverlapWithGt(current_topk, gt_row, args.top_k);
                            stop_recall = static_cast<float>(stop_hits_gt) /
                                          static_cast<float>(args.top_k);
                            delta_recall = full_recall - stop_recall;
                        }

                        out << global_query << ','
                            << top_l << ','
                            << stage_index << ','
                            << prefix << ','
                            << (ComputeTopKOverlap(current_topk, oracle_topk_ids) == args.top_k ? 0 : 1)
                            << ','
                            << std::log2(static_cast<double>(top_l)) << ','
                            << (static_cast<double>(prefix) / static_cast<double>(top_l)) << ','
                            << current_kth << ','
                            << delta_kth_last << ','
                            << previous_delta_kth << ','
                            << topk_changed_flag << ','
                            << entered_topk_ratio << ','
                            << current_topk_spread << ','
                            << (next_probe_end == prefix
                                    ? 0.0
                                    : static_cast<double>(next_window_live_count) /
                                          static_cast<double>(next_probe_end - prefix))
                            << ','
                            << (std::isfinite(best_live_lb) ? best_live_lb - current_kth : 0.0f)
                            << ','
                            << (window_size == 0
                                    ? 0.0
                                    : static_cast<double>(window_exact) /
                                          static_cast<double>(window_size))
                            << ','
                            << (window_size == 0
                                    ? 0.0
                                    : static_cast<double>(window_filtered) /
                                          static_cast<double>(window_size))
                            << ','
                            << (prefix == 0 ? 0.0
                                            : static_cast<double>(exact_count) /
                                                  static_cast<double>(prefix))
                            << ','
                            << exact_count << ','
                            << filtered_count << ','
                            << next_window_live_count << ','
                            << (next_probe_end - prefix) << ','
                            << ComputeTopKOverlap(current_topk, oracle_topk_ids) << ','
                            << stop_hits_gt << ','
                            << full_hits_gt << ','
                            << stop_recall << ','
                            << full_recall << ','
                            << delta_recall
                            << '\n';

                        previous_checkpoint_kth = current_kth;
                        previous_delta_kth = delta_kth_last;
                    }
                }

                out.flush();
                std::cout << "[learned_rerank_stop_dataset] top_l=" << top_l
                          << " finished queries " << (query_offset + batch_queries)
                          << " / " << args.num_queries << std::endl;
            }
        }

        std::cout << "[learned_rerank_stop_dataset] wrote " << args.output_csv << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_learned_rerank_stop_dataset] " << e.what() << std::endl;
        return 1;
    }
}
