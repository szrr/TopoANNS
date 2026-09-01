#include "topoanns/bam_vector_provider.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kPageSize = 4096;

struct Options {
    std::filesystem::path trace_path;
    std::filesystem::path payload_file;
    std::filesystem::path depth_histogram_path;
    std::filesystem::path controller_path = "/dev/libnvm0";
    std::size_t header_bytes = 0;
    std::size_t payload_bytes = 0;
    std::size_t device_offset_bytes = 0;
    std::size_t cache_bytes = 16ULL << 20;
    std::size_t queue_depth = 128;
    std::size_t num_queues = 32;
    std::size_t blocks = 2048;
    std::size_t warps_per_block = 4;
    std::size_t warmup_requests = 100000;
    std::size_t measure_requests = 0;
    std::size_t repetitions = 5;
    std::uint32_t cuda_device = 0;
    bool profile_depth = false;
    topoanns::BamTraceRequestMapping request_mapping =
        topoanns::BamTraceRequestMapping::kWarpPerPage;
};

std::size_t ParseSize(const std::string& value, const char* name) {
    std::size_t consumed = 0;
    const unsigned long long parsed = std::stoull(value, &consumed, 0);
    if (consumed != value.size() ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
    }
    return static_cast<std::size_t>(parsed);
}

Options ParseOptions(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--profile-depth") {
            options.profile_depth = true;
            continue;
        }
        if (i + 1 >= argc) {
            throw std::invalid_argument("missing value for " + key);
        }
        const std::string value = argv[++i];
        if (key == "--trace") {
            options.trace_path = value;
        } else if (key == "--payload-file") {
            options.payload_file = value;
        } else if (key == "--depth-histogram") {
            options.depth_histogram_path = value;
        } else if (key == "--controller") {
            options.controller_path = value;
        } else if (key == "--header-bytes") {
            options.header_bytes = ParseSize(value, "header bytes");
        } else if (key == "--payload-bytes") {
            options.payload_bytes = ParseSize(value, "payload bytes");
        } else if (key == "--device-offset-bytes") {
            options.device_offset_bytes = ParseSize(value, "device offset");
        } else if (key == "--cache-bytes") {
            options.cache_bytes = ParseSize(value, "cache bytes");
        } else if (key == "--queue-depth") {
            options.queue_depth = ParseSize(value, "queue depth");
        } else if (key == "--num-queues") {
            options.num_queues = ParseSize(value, "num queues");
        } else if (key == "--blocks") {
            options.blocks = ParseSize(value, "blocks");
        } else if (key == "--warps-per-block") {
            options.warps_per_block = ParseSize(value, "warps per block");
        } else if (key == "--warmup-requests") {
            options.warmup_requests = ParseSize(value, "warmup requests");
        } else if (key == "--measure-requests") {
            options.measure_requests = ParseSize(value, "measure requests");
        } else if (key == "--repetitions") {
            options.repetitions = ParseSize(value, "repetitions");
        } else if (key == "--cuda-device") {
            options.cuda_device = static_cast<std::uint32_t>(ParseSize(value, "CUDA device"));
        } else if (key == "--request-mapping") {
            if (value == "warp") {
                options.request_mapping =
                    topoanns::BamTraceRequestMapping::kWarpPerPage;
            } else if (value == "thread") {
                options.request_mapping =
                    topoanns::BamTraceRequestMapping::kThreadPerPage;
            } else {
                throw std::invalid_argument(
                    "--request-mapping must be warp or thread");
            }
        } else {
            throw std::invalid_argument("unknown argument: " + key);
        }
    }
    if (options.trace_path.empty() || options.payload_file.empty()) {
        throw std::invalid_argument("--trace and --payload-file are required");
    }
    return options;
}

std::vector<std::uint64_t> ReadTrace(const std::filesystem::path& path) {
    const std::uintmax_t bytes = std::filesystem::file_size(path);
    if (bytes == 0 || bytes % sizeof(std::uint64_t) != 0) {
        throw std::runtime_error("trace must contain packed uint64 page IDs");
    }
    std::vector<std::uint64_t> trace(bytes / sizeof(std::uint64_t));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(trace.data()), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("failed to read trace: " + path.string());
    }
    return trace;
}

std::uint64_t Delta(std::uint64_t after, std::uint64_t before) {
    return after >= before ? after - before : 0;
}

std::size_t DepthQuantile(const std::vector<std::uint64_t>& histogram,
                          double quantile) {
    std::uint64_t total = 0;
    for (std::uint64_t count : histogram) {
        total += count;
    }
    if (total == 0) {
        return 0;
    }
    const std::uint64_t target = static_cast<std::uint64_t>(quantile * (total - 1));
    std::uint64_t cumulative = 0;
    for (std::size_t depth = 0; depth < histogram.size(); ++depth) {
        cumulative += histogram[depth];
        if (cumulative > target) {
            return depth;
        }
    }
    return histogram.size() - 1;
}

void PrintDepth(const topoanns::BamIoProfileSnapshot& snapshot) {
    long double weighted = 0.0;
    std::uint64_t samples = 0;
    std::size_t max_depth = 0;
    for (std::size_t depth = 0; depth < snapshot.submit_depth_histogram.size(); ++depth) {
        const std::uint64_t count = snapshot.submit_depth_histogram[depth];
        weighted += static_cast<long double>(depth) * count;
        samples += count;
        if (count != 0) {
            max_depth = depth;
        }
    }
    const double mean = samples == 0 ? 0.0 : static_cast<double>(weighted / samples);
    std::cout << "[BAM_DEPTH] submissions=" << snapshot.profiled_submissions
              << " completions=" << snapshot.profiled_completions
              << " final_outstanding=" << snapshot.current_outstanding
              << " mean=" << mean
              << " p50=" << DepthQuantile(snapshot.submit_depth_histogram, 0.50)
              << " p90=" << DepthQuantile(snapshot.submit_depth_histogram, 0.90)
              << " p99=" << DepthQuantile(snapshot.submit_depth_histogram, 0.99)
              << " max=" << max_depth << '\n';
}

void WriteDepthHistogram(const std::filesystem::path& path,
                         const topoanns::BamIoProfileSnapshot& snapshot) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("failed to open depth histogram: " + path.string());
    }
    output << "outstanding_depth,submit_count\n";
    for (std::size_t depth = 0; depth < snapshot.submit_depth_histogram.size(); ++depth) {
        output << depth << ',' << snapshot.submit_depth_histogram[depth] << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = ParseOptions(argc, argv);
        std::vector<std::uint64_t> trace = ReadTrace(options.trace_path);
        const std::size_t payload_bytes = options.payload_bytes == 0
            ? std::filesystem::file_size(options.payload_file) - options.header_bytes
            : options.payload_bytes;
        if (payload_bytes == 0 || payload_bytes % kPageSize != 0) {
            throw std::runtime_error("payload size must be a nonzero multiple of 4096");
        }
        const std::uint64_t payload_pages = payload_bytes / kPageSize;
        const auto invalid = std::find_if(trace.begin(), trace.end(),
            [payload_pages](std::uint64_t page) { return page >= payload_pages; });
        if (invalid != trace.end()) {
            throw std::runtime_error("trace page ID exceeds payload page count");
        }
        if (options.warmup_requests >= trace.size()) {
            throw std::runtime_error("warmup consumes the entire trace");
        }
        const std::size_t available = trace.size() - options.warmup_requests;
        const std::size_t measure_requests = options.measure_requests == 0
            ? available : options.measure_requests;
        if (measure_requests > available || options.repetitions == 0) {
            throw std::runtime_error("invalid measurement request count or repetitions");
        }

        topoanns::BamVectorProviderOptions bam_options;
        bam_options.controller_path = options.controller_path;
        bam_options.device_offset_bytes = options.device_offset_bytes;
        bam_options.page_cache_size_bytes = options.cache_bytes;
        bam_options.cuda_device = options.cuda_device;
        bam_options.queue_depth = options.queue_depth;
        bam_options.num_queues = options.num_queues;
        bam_options.payload_bytes_override = payload_bytes;
        topoanns::BamVectorPageProvider provider(
            options.payload_file, options.header_bytes, kPageSize, bam_options);
        const auto device_trace =
            topoanns::CudaBuffer<std::uint64_t>::CopyFromHost(trace);

        const auto run_trace = [&](std::size_t start, std::size_t count) {
            return topoanns::RunBam4kTraceBenchmark(
                provider, device_trace, start, count, options.blocks,
                options.warps_per_block, kPageSize, options.request_mapping);
        };

        if (options.warmup_requests != 0) {
            provider.ResetIoDepthProfile(false);
            run_trace(0, options.warmup_requests);
        }

        std::cout << std::fixed << std::setprecision(3);
        const char* mapping =
            options.request_mapping == topoanns::BamTraceRequestMapping::kThreadPerPage
                ? "thread"
                : "warp";
        for (std::size_t repetition = 0; repetition < options.repetitions; ++repetition) {
            provider.ResetIoDepthProfile(options.profile_depth);
            const topoanns::BamIoProfileSnapshot before = provider.ReadIoProfile();
            const topoanns::BamTraceBenchmarkResult run =
                run_trace(options.warmup_requests, measure_requests);
            if (repetition == 0) {
                std::set<std::uint32_t> unique_sms;
                std::set<std::uint32_t> unique_queues;
                std::cout << "[BAM_BLOCK_MAP] sm_ids=";
                for (std::size_t i = 0; i < run.block_sm_ids.size(); ++i) {
                    if (i != 0) {
                        std::cout << ',';
                    }
                    const std::uint32_t sm_id = run.block_sm_ids[i];
                    std::cout << sm_id;
                    unique_sms.insert(sm_id);
                    unique_queues.insert(
                        sm_id % static_cast<std::uint32_t>(options.num_queues));
                }
                std::cout << " unique_sms=" << unique_sms.size()
                          << " unique_queues=" << unique_queues.size()
                          << " queue_ids=";
                bool first = true;
                for (const std::uint32_t queue_id : unique_queues) {
                    if (!first) {
                        std::cout << ',';
                    }
                    std::cout << queue_id;
                    first = false;
                }
                std::cout << '\n';
            }
            const topoanns::BamIoProfileSnapshot after = provider.ReadIoProfile();
            const std::uint64_t physical_reads =
                Delta(after.physical_reads, before.physical_reads);
            const std::uint64_t gpu_hits =
                Delta(after.gpu_cache_hits, before.gpu_cache_hits);
            const std::uint64_t host_hits =
                Delta(after.host_cache_hits, before.host_cache_hits);
            const double seconds = run.elapsed_ms / 1000.0;
            const double logical_iops = measure_requests / seconds;
            const double physical_iops = physical_reads / seconds;
            const double gib_per_second =
                physical_reads * static_cast<double>(kPageSize) /
                seconds / static_cast<double>(1ULL << 30);
            std::cout << "[BAM_4K_RESULT] mapping=" << mapping
                      << " blocks=" << options.blocks
                      << " threads_per_block=" << options.warps_per_block * 32
                      << " rep=" << repetition
                      << " logical_requests=" << measure_requests
                      << " physical_reads=" << physical_reads
                      << " gpu_hits=" << gpu_hits
                      << " host_hits=" << host_hits
                      << " elapsed_ms=" << run.elapsed_ms
                      << " logical_iops=" << logical_iops
                      << " physical_iops=" << physical_iops
                      << " gib_s=" << gib_per_second
                      << " checksum=" << run.checksum << '\n';
            if (options.profile_depth) {
                PrintDepth(after);
                if (!options.depth_histogram_path.empty()) {
                    WriteDepthHistogram(options.depth_histogram_path, after);
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "topoanns_bam_4k_trace_bench: " << error.what() << '\n';
        return 1;
    }
}
