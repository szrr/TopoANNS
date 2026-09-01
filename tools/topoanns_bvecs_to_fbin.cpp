#include "topoanns/bvecs_io.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t kProgressStepVectors = 10'000'000ULL;

struct Args {
    std::filesystem::path input_bvecs;
    std::filesystem::path output_fbin;
    std::uint64_t num_vectors = 0;
    std::size_t chunk_vectors = 65536;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_bvecs_to_fbin"
        << " --input-bvecs <path>"
        << " --output-fbin <path>"
        << " [--num-vectors <count>]"
        << " [--chunk-vectors <count>]"
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
        if (flag == "--input-bvecs") {
            args.input_bvecs = read_value("--input-bvecs");
        } else if (flag == "--output-fbin") {
            args.output_fbin = read_value("--output-fbin");
        } else if (flag == "--num-vectors") {
            args.num_vectors = std::stoull(read_value("--num-vectors"));
        } else if (flag == "--chunk-vectors") {
            args.chunk_vectors = std::stoull(read_value("--chunk-vectors"));
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }
    if (args.input_bvecs.empty() || args.output_fbin.empty()) {
        Usage();
    }
    if (args.chunk_vectors == 0) {
        throw std::runtime_error("chunk-vectors must be > 0.");
    }
    return args;
}

int Main(const Args& args) {
    const topoanns::BvecsMetadata metadata = topoanns::ReadBvecsMetadata(args.input_bvecs);
    const std::uint64_t total_vectors =
        args.num_vectors == 0 ? metadata.num_vectors : args.num_vectors;
    if (total_vectors > metadata.num_vectors) {
        throw std::runtime_error("num-vectors exceeds input bvecs rows.");
    }

    std::ifstream in(args.input_bvecs, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open input bvecs: " + args.input_bvecs.string());
    }
    std::ofstream out(args.output_fbin, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open output fbin: " + args.output_fbin.string());
    }

    const std::uint32_t out_rows = static_cast<std::uint32_t>(total_vectors);
    const std::uint32_t out_dim = metadata.dim;
    out.write(reinterpret_cast<const char*>(&out_rows), sizeof(out_rows));
    out.write(reinterpret_cast<const char*>(&out_dim), sizeof(out_dim));

    std::vector<std::uint8_t> raw_records(args.chunk_vectors * metadata.record_bytes, 0);
    std::vector<float> float_chunk(args.chunk_vectors * metadata.dim, 0.0f);

    std::uint64_t processed = 0;
    while (processed < total_vectors) {
        const std::uint64_t current =
            std::min<std::uint64_t>(args.chunk_vectors, total_vectors - processed);
        const std::size_t bytes_to_read =
            static_cast<std::size_t>(current) * metadata.record_bytes;
        in.read(reinterpret_cast<char*>(raw_records.data()),
                static_cast<std::streamsize>(bytes_to_read));
        if (in.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
            throw std::runtime_error("Short read while converting bvecs to fbin.");
        }

        for (std::uint64_t i = 0; i < current; ++i) {
            const std::uint8_t* record =
                raw_records.data() + i * metadata.record_bytes;
            std::uint32_t record_dim = 0;
            std::memcpy(&record_dim, record, sizeof(record_dim));
            if (record_dim != metadata.dim) {
                throw std::runtime_error("Per-record dimension mismatch in bvecs.");
            }
            const std::uint8_t* payload = record + sizeof(std::uint32_t);
            float* dst = float_chunk.data() + static_cast<std::size_t>(i) * metadata.dim;
            for (std::uint32_t d = 0; d < metadata.dim; ++d) {
                dst[d] = static_cast<float>(payload[d]);
            }
        }

        out.write(reinterpret_cast<const char*>(float_chunk.data()),
                  static_cast<std::streamsize>(current * metadata.dim * sizeof(float)));
        processed += current;
        if (processed == total_vectors ||
            processed / kProgressStepVectors !=
                (processed - current) / kProgressStepVectors) {
            std::cout << "[topoanns_bvecs_to_fbin] converted "
                      << processed << " / " << total_vectors << std::endl;
        }
    }

    std::cout << "[topoanns_bvecs_to_fbin] dim=" << metadata.dim
              << " rows=" << total_vectors
              << " output=" << args.output_fbin << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(ParseArgs(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_bvecs_to_fbin] " << e.what() << std::endl;
        return 1;
    }
}
