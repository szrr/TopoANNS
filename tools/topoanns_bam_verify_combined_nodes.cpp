#include "topoanns/bam_runtime_config.hpp"
#include "topoanns/bam_vector_provider.hpp"
#include "topoanns/diskann_disk_index.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Args {
    std::filesystem::path disk_index;
    std::optional<std::filesystem::path> bam_config_path;
    topoanns::BamRuntimeConfigOverrides bam_overrides;
    bool allow_bam_controller_override = false;
    std::uint64_t device_offset_bytes = 0;
    std::size_t sample_count = 64;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_bam_verify_combined_nodes"
        << " --disk-index <path> --device-offset-bytes <bytes>"
        << " [--samples <count>] [--bam-config-path <path>]"
        << " [--allow-bam-controller-override] [--controller-path <path>]"
        << " [--page-cache-bytes <bytes>] [--queue-depth <count>]"
        << " [--num-queues <count>] [--cuda-device <id>]"
        << std::endl;
    std::exit(1);
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
        if (flag == "--disk-index") {
            args.disk_index = read_value("--disk-index");
        } else if (flag == "--device-offset-bytes") {
            args.device_offset_bytes =
                std::stoull(read_value("--device-offset-bytes"));
        } else if (flag == "--samples") {
            args.sample_count = std::stoull(read_value("--samples"));
        } else if (flag == "--bam-config-path") {
            args.bam_config_path = read_value("--bam-config-path");
        } else if (flag == "--allow-bam-controller-override") {
            args.allow_bam_controller_override = true;
        } else if (flag == "--controller-path") {
            args.bam_overrides.controller_path = read_value("--controller-path");
        } else if (flag == "--page-cache-bytes") {
            args.bam_overrides.page_cache_size_bytes =
                std::stoull(read_value("--page-cache-bytes"));
        } else if (flag == "--queue-depth") {
            args.bam_overrides.queue_depth =
                std::stoull(read_value("--queue-depth"));
        } else if (flag == "--num-queues") {
            args.bam_overrides.num_queues =
                std::stoull(read_value("--num-queues"));
        } else if (flag == "--cuda-device") {
            args.bam_overrides.cuda_device =
                static_cast<std::uint32_t>(
                    std::stoul(read_value("--cuda-device")));
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }
    if (args.disk_index.empty() || args.sample_count == 0) {
        Usage();
    }
    return args;
}

std::vector<std::uint32_t> BuildSampleIds(std::uint64_t num_nodes,
                                          std::uint64_t medoid,
                                          std::size_t sample_count,
                                          std::size_t nodes_per_page) {
    std::vector<std::uint32_t> ids;
    const auto add = [&](std::uint64_t id) {
        if (id < num_nodes &&
            std::find(ids.begin(), ids.end(), static_cast<std::uint32_t>(id)) ==
                ids.end()) {
            ids.push_back(static_cast<std::uint32_t>(id));
        }
    };
    add(0);
    add(1);
    add(nodes_per_page - 1);
    add(nodes_per_page);
    add(medoid);
    add(num_nodes / 5 - 1);
    add(num_nodes / 5);
    add(num_nodes / 2);
    add(num_nodes - 1);

    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    while (ids.size() < std::min<std::uint64_t>(sample_count, num_nodes)) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        add((state * 0x2545f4914f6cdd1dULL) % num_nodes);
    }
    if (ids.size() > sample_count) {
        ids.resize(sample_count);
    }
    return ids;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        const topoanns::DiskannDiskIndexLayout layout =
            topoanns::DiskannDiskIndexLayout::Load(args.disk_index);
        const std::size_t page_size = topoanns::kDefaultPageSizeBytes;
        const std::uint64_t file_bytes =
            std::filesystem::file_size(args.disk_index);
        if (file_bytes <= page_size || (file_bytes - page_size) % page_size != 0) {
            throw std::runtime_error(
                "DiskANN payload after the metadata page must be 4KB aligned.");
        }
        if (layout.metadata().num_nodes >
            std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("Verifier supports at most uint32 node IDs.");
        }
        const std::size_t node_bytes = layout.metadata().max_node_len;
        const std::size_t nodes_per_page =
            layout.metadata().nodes_per_sector;
        const auto node_ids = BuildSampleIds(
            layout.metadata().num_nodes, layout.metadata().medoid_id,
            args.sample_count, nodes_per_page);

        const topoanns::BamRuntimeConfig bam_runtime =
            topoanns::ResolveBamRuntimeConfig(
                args.bam_config_path, args.bam_overrides,
                args.allow_bam_controller_override);
        topoanns::ThrowIfCudaError(
            cudaSetDevice(static_cast<int>(bam_runtime.cuda_device)),
            "cudaSetDevice");
        topoanns::BamVectorProviderOptions options;
        topoanns::ApplyBamRuntimeConfig(bam_runtime, &options);
        options.device_offset_bytes = args.device_offset_bytes;
        options.payload_bytes_override = file_bytes - page_size;
        topoanns::BamVectorPageProvider provider(
            args.disk_index, page_size, page_size, options);

        const std::vector<std::uint8_t> actual =
            topoanns::ReadBamCombinedNodeRecords(
                provider, node_ids, node_bytes, nodes_per_page);
        std::ifstream input(args.disk_index, std::ios::binary);
        if (!input.is_open()) {
            throw std::runtime_error("Failed to open source DiskANN index.");
        }
        std::vector<std::uint8_t> expected(node_bytes);
        for (std::size_t sample = 0; sample < node_ids.size(); ++sample) {
            input.seekg(static_cast<std::streamoff>(
                            layout.NodeOffsetBytes(node_ids[sample])),
                        std::ios::beg);
            input.read(reinterpret_cast<char*>(expected.data()),
                       static_cast<std::streamsize>(expected.size()));
            if (!input.good()) {
                throw std::runtime_error(
                    "Short source read for node " +
                    std::to_string(node_ids[sample]));
            }
            const std::uint8_t* observed =
                actual.data() + sample * node_bytes;
            if (std::memcmp(expected.data(), observed, node_bytes) != 0) {
                const auto mismatch = std::mismatch(
                    expected.begin(), expected.end(), observed);
                throw std::runtime_error(
                    "Combined record mismatch at node " +
                    std::to_string(node_ids[sample]) + ", byte " +
                    std::to_string(
                        std::distance(expected.begin(), mismatch.first)));
            }
        }
        std::cout << "[topoanns_bam_verify_combined_nodes]"
                  << " verified=" << node_ids.size()
                  << " num_nodes=" << layout.metadata().num_nodes
                  << " node_bytes=" << node_bytes
                  << " nodes_per_page=" << nodes_per_page
                  << " device_offset_bytes=" << args.device_offset_bytes
                  << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_bam_verify_combined_nodes] "
                  << e.what() << std::endl;
        return 1;
    }
}
