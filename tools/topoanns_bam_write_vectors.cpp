#include "topoanns/bam_runtime_config.hpp"
#include "topoanns/bam_vector_provider.hpp"
#include "topoanns/vector_store_builder.hpp"

#include <cuda_runtime.h>

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct Args {
    std::filesystem::path vector_store_path;
    std::optional<std::filesystem::path> bam_config_path;
    topoanns::BamRuntimeConfigOverrides bam_overrides;
    bool allow_bam_controller_override = false;
    std::size_t device_offset_bytes = 0;
    bool dry_run = false;
};

void RequireL40Device(std::uint32_t device_id, const char* context) {
    cudaDeviceProp props{};
    const cudaError_t status =
        cudaGetDeviceProperties(&props, static_cast<int>(device_id));
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(context) +
                                 ": cudaGetDeviceProperties failed for requested BAM GPU.");
    }
    const std::string gpu_name(props.name);
    if (gpu_name.find("L40") == std::string::npos) {
        throw std::runtime_error(std::string(context) +
                                 ": BAM path must run on an NVIDIA L40, but the selected CUDA "
                                 "device is \"" + gpu_name + "\" (logical cuda:" +
                                 std::to_string(device_id) + ").");
    }
}

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_bam_write_vectors"
        << " --vector-store <path>"
        << " [--bam-config-path <path>]"
        << " [--allow-bam-controller-override]"
        << " [--controller-path <path>]"
        << " [--device-offset-bytes <bytes>]"
        << " [--page-cache-bytes <bytes>]"
        << " [--queue-depth <count>]"
        << " [--num-queues <count>]"
        << " [--cuda-device <id>]"
        << " [--dry-run]"
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
        if (flag == "--vector-store") {
            args.vector_store_path = read_value("--vector-store");
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
        } else if (flag == "--dry-run") {
            args.dry_run = true;
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }

    if (args.vector_store_path.empty()) {
        Usage();
    }
    return args;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        const topoanns::BamRuntimeConfig bam_runtime = topoanns::ResolveBamRuntimeConfig(
            args.bam_config_path, args.bam_overrides, args.allow_bam_controller_override);
        const topoanns::VectorStoreHeader header =
            topoanns::VectorStoreBuilder::ReadHeader(args.vector_store_path);
        const std::uint64_t file_bytes = std::filesystem::file_size(args.vector_store_path);
        const std::uint64_t payload_bytes = file_bytes - sizeof(topoanns::VectorStoreHeader);
        const std::uint64_t payload_pages = payload_bytes / header.page_size_bytes;

        std::cout << "[topoanns_bam_write_vectors] vector_store=" << args.vector_store_path
                  << " controller=" << bam_runtime.controller_path
                  << " device_offset_bytes=" << args.device_offset_bytes
                  << " payload_bytes=" << payload_bytes
                  << " payload_pages=" << payload_pages
                  << " page_cache_bytes=" << bam_runtime.page_cache_size_bytes
                  << " queue_depth=" << bam_runtime.queue_depth
                  << " num_queues=" << bam_runtime.num_queues
                  << " cuda_device=" << bam_runtime.cuda_device
                  << " dry_run=" << (args.dry_run ? 1 : 0)
                  << std::endl;

        if (args.dry_run) {
            return 0;
        }

        const cudaError_t device_status =
            cudaSetDevice(static_cast<int>(bam_runtime.cuda_device));
        if (device_status != cudaSuccess) {
            throw std::runtime_error("cudaSetDevice failed for requested BAM GPU.");
        }
        RequireL40Device(bam_runtime.cuda_device, "topoanns_bam_write_vectors");

        topoanns::BamVectorProviderOptions options;
        topoanns::ApplyBamRuntimeConfig(bam_runtime, &options);
        options.device_offset_bytes = args.device_offset_bytes;
        topoanns::WriteVectorStorePayloadToBam(args.vector_store_path,
                                               sizeof(topoanns::VectorStoreHeader),
                                               static_cast<std::size_t>(header.page_size_bytes),
                                               options);
        std::cout << "[topoanns_bam_write_vectors] completed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_bam_write_vectors] " << e.what() << std::endl;
        return 1;
    }
}
