#include "topoanns/bam_runtime_config.hpp"
#include "topoanns/bam_vector_provider.hpp"
#include "topoanns/diskann_disk_index.hpp"
#include "topoanns/topology_layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    std::filesystem::path disk_index;
    std::filesystem::path topology;
    std::optional<std::filesystem::path> bam_config_path;
    topoanns::BamRuntimeConfigOverrides bam_overrides;
    bool allow_bam_controller_override = false;
    std::uint64_t device_offset_bytes = 0;
    std::size_t verify_samples = 4096;
    bool dry_run = false;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_bam_write_combined_nodes"
        << " --disk-index <path> --topology <path> --device-offset-bytes <bytes>"
        << " [--verify-samples <count>] [--dry-run]"
        << " [--bam-config-path <path>] [--allow-bam-controller-override]"
        << " [--controller-path <path>] [--page-cache-bytes <bytes>]"
        << " [--queue-depth <count>] [--num-queues <count>] [--cuda-device <id>]"
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
        } else if (flag == "--topology") {
            args.topology = read_value("--topology");
        } else if (flag == "--bam-config-path") {
            args.bam_config_path = read_value("--bam-config-path");
        } else if (flag == "--allow-bam-controller-override") {
            args.allow_bam_controller_override = true;
        } else if (flag == "--controller-path") {
            args.bam_overrides.controller_path = read_value("--controller-path");
        } else if (flag == "--device-offset-bytes") {
            args.device_offset_bytes = std::stoull(read_value("--device-offset-bytes"));
        } else if (flag == "--page-cache-bytes") {
            args.bam_overrides.page_cache_size_bytes =
                std::stoull(read_value("--page-cache-bytes"));
        } else if (flag == "--queue-depth") {
            args.bam_overrides.queue_depth = std::stoull(read_value("--queue-depth"));
        } else if (flag == "--num-queues") {
            args.bam_overrides.num_queues = std::stoull(read_value("--num-queues"));
        } else if (flag == "--cuda-device") {
            args.bam_overrides.cuda_device =
                static_cast<std::uint32_t>(std::stoul(read_value("--cuda-device")));
        } else if (flag == "--verify-samples") {
            args.verify_samples = std::stoull(read_value("--verify-samples"));
        } else if (flag == "--dry-run") {
            args.dry_run = true;
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }
    if (args.disk_index.empty() || args.topology.empty()) {
        Usage();
    }
    return args;
}

void RequireL40Device(std::uint32_t device_id) {
    cudaDeviceProp props{};
    const cudaError_t status = cudaGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != cudaSuccess) {
        throw std::runtime_error("cudaGetDeviceProperties failed for requested BaM GPU.");
    }
    if (std::string(props.name).find("L40") == std::string::npos) {
        throw std::runtime_error("Combined-node BaM write must run on an NVIDIA L40.");
    }
}

std::vector<std::uint64_t> BuildSampleIds(std::uint64_t num_nodes,
                                          std::uint64_t medoid,
                                          std::size_t sample_count) {
    std::vector<std::uint64_t> ids;
    if (num_nodes == 0 || sample_count == 0) {
        return ids;
    }
    ids.reserve(sample_count + 3);
    ids.push_back(0);
    ids.push_back(num_nodes - 1);
    if (medoid < num_nodes) {
        ids.push_back(medoid);
    }
    std::uint64_t state = 0x9e3779b97f4a7c15ULL;
    while (ids.size() < sample_count) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        ids.push_back((state * 0x2545f4914f6cdd1dULL) % num_nodes);
    }
    return ids;
}

void VerifyTopologySamples(const std::filesystem::path& disk_index,
                           const std::filesystem::path& topology,
                           const topoanns::DiskannDiskIndexLayout& layout,
                           std::size_t sample_count) {
    const topoanns::TopologyHeader topology_header =
        topoanns::TopologyLayout::ReadHeader(topology);
    if (topology_header.num_nodes != layout.metadata().num_nodes ||
        topology_header.degree != topoanns::kFixedTopologyDegree ||
        layout.neighbor_capacity() < topoanns::kFixedTopologyDegree) {
        throw std::runtime_error("DiskANN and TopoANNS topology metadata do not match.");
    }

    std::ifstream disk_in(disk_index, std::ios::binary);
    std::ifstream topology_in(topology, std::ios::binary);
    if (!disk_in.is_open() || !topology_in.is_open()) {
        throw std::runtime_error("Failed to open DiskANN index or topology for verification.");
    }

    std::vector<char> node(static_cast<std::size_t>(layout.metadata().max_node_len));
    std::array<std::uint32_t, topoanns::kFixedTopologyDegree> expected{};
    std::array<std::uint32_t, topoanns::kFixedTopologyDegree> actual{};
    const std::size_t coord_bytes = layout.coord_bytes();
    const auto ids = BuildSampleIds(layout.metadata().num_nodes,
                                    layout.metadata().medoid_id,
                                    sample_count);
    for (const std::uint64_t node_id : ids) {
        disk_in.seekg(static_cast<std::streamoff>(layout.NodeOffsetBytes(node_id)),
                      std::ios::beg);
        disk_in.read(node.data(), static_cast<std::streamsize>(node.size()));
        topology_in.seekg(static_cast<std::streamoff>(
                              topoanns::TopologyLayout::NodeOffsetBytes(node_id)),
                          std::ios::beg);
        topology_in.read(reinterpret_cast<char*>(actual.data()),
                         static_cast<std::streamsize>(sizeof(actual)));
        if (!disk_in.good() || !topology_in.good()) {
            throw std::runtime_error("Short read during combined-node topology verification.");
        }

        expected.fill(topoanns::kInvalidNodeId);
        std::uint32_t degree = 0;
        std::memcpy(&degree, node.data() + coord_bytes, sizeof(degree));
        const auto* neighbors = reinterpret_cast<const std::uint32_t*>(
            node.data() + coord_bytes + sizeof(std::uint32_t));
        const std::size_t limit =
            std::min<std::size_t>(degree, topoanns::kFixedTopologyDegree);
        for (std::size_t i = 0; i < limit; ++i) {
            if (neighbors[i] < layout.metadata().num_nodes) {
                expected[i] = neighbors[i];
            }
        }
        if (expected != actual) {
            throw std::runtime_error("Topology mismatch at node " + std::to_string(node_id));
        }
    }
    std::cout << "[topoanns_combined_verify] verified_samples=" << ids.size() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        const topoanns::DiskannDiskIndexLayout layout =
            topoanns::DiskannDiskIndexLayout::Load(args.disk_index);
        const std::uint64_t file_bytes = std::filesystem::file_size(args.disk_index);
        if (file_bytes < topoanns::kDefaultPageSizeBytes) {
            throw std::runtime_error("DiskANN index is smaller than its metadata page.");
        }
        const std::uint64_t payload_bytes = file_bytes - topoanns::kDefaultPageSizeBytes;
        if (payload_bytes % topoanns::kDefaultPageSizeBytes != 0) {
            throw std::runtime_error("DiskANN payload is not 4KB page aligned.");
        }
        VerifyTopologySamples(args.disk_index, args.topology, layout, args.verify_samples);

        const topoanns::BamRuntimeConfig bam_runtime = topoanns::ResolveBamRuntimeConfig(
            args.bam_config_path, args.bam_overrides, args.allow_bam_controller_override);
        std::cout << "[topoanns_bam_write_combined_nodes] disk_index=" << args.disk_index
                  << " controller=" << bam_runtime.controller_path
                  << " device_offset_bytes=" << args.device_offset_bytes
                  << " payload_bytes=" << payload_bytes
                  << " payload_pages=" << payload_bytes / topoanns::kDefaultPageSizeBytes
                  << " num_nodes=" << layout.metadata().num_nodes
                  << " dim=" << layout.metadata().vector_dim
                  << " node_bytes=" << layout.metadata().max_node_len
                  << " nodes_per_page=" << layout.metadata().nodes_per_sector
                  << " dry_run=" << (args.dry_run ? 1 : 0) << std::endl;
        if (args.dry_run) {
            return 0;
        }

        topoanns::ThrowIfCudaError(cudaSetDevice(static_cast<int>(bam_runtime.cuda_device)),
                                   "cudaSetDevice");
        RequireL40Device(bam_runtime.cuda_device);
        topoanns::BamVectorProviderOptions options;
        topoanns::ApplyBamRuntimeConfig(bam_runtime, &options);
        options.device_offset_bytes = args.device_offset_bytes;
        topoanns::WriteVectorStorePayloadToBam(args.disk_index,
                                               topoanns::kDefaultPageSizeBytes,
                                               topoanns::kDefaultPageSizeBytes,
                                               options);
        std::cout << "[topoanns_bam_write_combined_nodes] completed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_bam_write_combined_nodes] " << e.what() << std::endl;
        return 1;
    }
}
