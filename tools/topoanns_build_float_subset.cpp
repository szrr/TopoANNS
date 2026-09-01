#include "topoanns/diskann_disk_index.hpp"
#include "topoanns/topology_layout.hpp"
#include "topoanns/vector_page_layout.hpp"
#include "topoanns/vector_store_builder.hpp"

#include <algorithm>
#include <array>
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

struct FloatBinMetadata {
    std::uint64_t num_vectors = 0;
    std::uint32_t dim = 0;
};

struct Args {
    std::filesystem::path base_fbin;
    std::filesystem::path disk_index;
    std::filesystem::path pq_pivots;
    std::filesystem::path pq_compressed;
    std::filesystem::path output_dir;
    std::uint64_t num_vectors = 0;
    std::size_t chunk_vectors = 65536;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_build_float_subset"
        << " --base-fbin <path>"
        << " --disk-index <path>"
        << " --pq-pivots <path>"
        << " --pq-compressed <path>"
        << " --output-dir <path>"
        << " --num-vectors <count>"
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
        if (flag == "--base-fbin") {
            args.base_fbin = read_value("--base-fbin");
        } else if (flag == "--disk-index") {
            args.disk_index = read_value("--disk-index");
        } else if (flag == "--pq-pivots") {
            args.pq_pivots = read_value("--pq-pivots");
        } else if (flag == "--pq-compressed") {
            args.pq_compressed = read_value("--pq-compressed");
        } else if (flag == "--output-dir") {
            args.output_dir = read_value("--output-dir");
        } else if (flag == "--num-vectors") {
            args.num_vectors = std::stoull(read_value("--num-vectors"));
        } else if (flag == "--chunk-vectors") {
            args.chunk_vectors = std::stoull(read_value("--chunk-vectors"));
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }
    if (args.base_fbin.empty() || args.disk_index.empty() || args.pq_pivots.empty() ||
        args.pq_compressed.empty() || args.output_dir.empty() || args.num_vectors == 0) {
        Usage();
    }
    return args;
}

FloatBinMetadata ReadFloatBinMetadata(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open float bin: " + path.string());
    }
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!in.good() || rows <= 0 || cols <= 0) {
        throw std::runtime_error("Invalid float bin metadata in " + path.string());
    }
    return FloatBinMetadata{
        static_cast<std::uint64_t>(rows),
        static_cast<std::uint32_t>(cols),
    };
}

topoanns::VectorStoreHeader BuildVectorStoreHeader(const topoanns::VectorPageLayout& layout,
                                                   std::uint32_t dim,
                                                   std::uint64_t num_vectors) {
    topoanns::VectorStoreHeader header{};
    constexpr char kMagic[8] = {'T', 'P', 'V', 'E', 'C', 'T', 'O', 'R'};
    std::memcpy(header.magic, kMagic, sizeof(header.magic));
    header.version = topoanns::VectorStoreBuilder::kVersion;
    header.scalar_kind = static_cast<std::uint32_t>(topoanns::ScalarKind::kFloat32);
    header.dim = dim;
    header.num_vectors = num_vectors;
    header.page_size_bytes = layout.page_size_bytes();
    header.vector_bytes = layout.vector_bytes();
    header.vectors_per_page = layout.vectors_per_page();
    return header;
}

topoanns::TopologyHeader BuildTopologyHeader(std::uint64_t num_nodes) {
    topoanns::TopologyHeader header{};
    constexpr char kMagic[8] = {'T', 'O', 'P', 'O', 'A', 'N', 'N', 'S'};
    std::memcpy(header.magic, kMagic, sizeof(header.magic));
    header.version = topoanns::TopologyLayout::kVersion;
    header.degree = topoanns::kFixedTopologyDegree;
    header.num_nodes = num_nodes;
    header.header_bytes = sizeof(topoanns::TopologyHeader);
    header.payload_bytes =
        num_nodes * topoanns::kFixedTopologyDegree * sizeof(std::uint32_t);
    return header;
}

void CopyFile(const std::filesystem::path& src, const std::filesystem::path& dst) {
    std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
}

void SlicePqCompressed(const std::filesystem::path& src,
                       const std::filesystem::path& dst,
                       std::uint64_t num_vectors) {
    std::ifstream in(src, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open PQ compressed file: " + src.string());
    }
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!in.good() || rows < 0 || cols <= 0) {
        throw std::runtime_error("Invalid PQ compressed header: " + src.string());
    }
    if (num_vectors > static_cast<std::uint64_t>(rows)) {
        throw std::runtime_error("Requested PQ slice is larger than source rows.");
    }

    std::ofstream out(dst, std::ios::binary | std::ios::trunc);
    const std::int32_t out_rows = static_cast<std::int32_t>(num_vectors);
    out.write(reinterpret_cast<const char*>(&out_rows), sizeof(out_rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    constexpr std::size_t kBufferBytes = 16 * 1024 * 1024;
    std::vector<char> buffer(kBufferBytes);
    std::uint64_t remaining = num_vectors * static_cast<std::uint64_t>(cols);
    while (remaining > 0) {
        const std::size_t to_read =
            static_cast<std::size_t>(std::min<std::uint64_t>(remaining, buffer.size()));
        in.read(buffer.data(), static_cast<std::streamsize>(to_read));
        if (in.gcount() != static_cast<std::streamsize>(to_read)) {
            throw std::runtime_error("Short read while slicing PQ compressed file.");
        }
        out.write(buffer.data(), static_cast<std::streamsize>(to_read));
        remaining -= to_read;
    }
}

void ConvertVectors(const Args& args,
                    const FloatBinMetadata& metadata,
                    const std::filesystem::path& vector_store_path) {
    const topoanns::VectorPageLayout layout =
        topoanns::VectorPageLayout::Create(metadata.dim, topoanns::ScalarKind::kFloat32);
    const topoanns::VectorStoreHeader header =
        BuildVectorStoreHeader(layout, metadata.dim, args.num_vectors);

    std::ifstream in(args.base_fbin, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open base float bin: " + args.base_fbin.string());
    }
    in.seekg(static_cast<std::streamoff>(sizeof(std::int32_t) * 2), std::ios::beg);
    std::ofstream store_out(vector_store_path, std::ios::binary | std::ios::trunc);
    store_out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const std::size_t vector_bytes = static_cast<std::size_t>(metadata.dim) * sizeof(float);
    std::vector<std::uint8_t> raw_vectors(args.chunk_vectors * vector_bytes, 0);
    std::vector<std::uint8_t> page(layout.page_size_bytes(), 0);
    std::size_t page_fill = 0;

    std::uint64_t processed = 0;
    while (processed < args.num_vectors) {
        const std::uint64_t current =
            std::min<std::uint64_t>(args.chunk_vectors, args.num_vectors - processed);
        const std::size_t bytes_to_read = static_cast<std::size_t>(current) * vector_bytes;
        in.read(reinterpret_cast<char*>(raw_vectors.data()),
                static_cast<std::streamsize>(bytes_to_read));
        if (in.gcount() != static_cast<std::streamsize>(bytes_to_read)) {
            throw std::runtime_error("Short read while converting base vectors.");
        }

        for (std::uint64_t i = 0; i < current; ++i) {
            const std::uint8_t* vector_src = raw_vectors.data() + i * vector_bytes;
            std::memcpy(page.data() + page_fill * vector_bytes, vector_src, vector_bytes);
            ++page_fill;
            if (page_fill == layout.vectors_per_page()) {
                store_out.write(reinterpret_cast<const char*>(page.data()),
                                static_cast<std::streamsize>(page.size()));
                std::fill(page.begin(), page.end(), 0U);
                page_fill = 0;
            }
        }

        const std::uint64_t previous = processed;
        processed += current;
        if (processed == args.num_vectors ||
            processed / kProgressStepVectors != previous / kProgressStepVectors) {
            std::cout << "[topoanns_build_float] converted vectors " << processed << " / "
                      << args.num_vectors << std::endl;
        }
    }

    if (page_fill > 0) {
        store_out.write(reinterpret_cast<const char*>(page.data()),
                        static_cast<std::streamsize>(page.size()));
    }
}

void ExtractTopology(const Args& args,
                     const topoanns::DiskannDiskIndexLayout& layout,
                     const std::filesystem::path& topology_path) {
    std::ifstream in(args.disk_index, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open DiskANN index: " + args.disk_index.string());
    }
    std::ofstream out(topology_path, std::ios::binary | std::ios::trunc);
    const topoanns::TopologyHeader header = BuildTopologyHeader(args.num_vectors);
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    const std::size_t coord_bytes = layout.coord_bytes();
    const std::size_t neighbor_capacity = layout.neighbor_capacity();
    if (neighbor_capacity < topoanns::kFixedTopologyDegree) {
        throw std::runtime_error("DiskANN index has fewer than 64 neighbors per node.");
    }

    std::array<std::uint32_t, topoanns::kFixedTopologyDegree> neighbors{};
    neighbors.fill(topoanns::kInvalidNodeId);
    std::uint64_t filtered_neighbors = 0;
    std::uint64_t processed = 0;

    if (layout.metadata().nodes_per_sector > 0) {
        std::vector<char> sector(topoanns::kDefaultPageSizeBytes, 0);
        const std::uint64_t total_sectors =
            (args.num_vectors + layout.metadata().nodes_per_sector - 1) /
            layout.metadata().nodes_per_sector;
        in.seekg(static_cast<std::streamoff>(topoanns::kDefaultPageSizeBytes), std::ios::beg);
        for (std::uint64_t sector_id = 0; sector_id < total_sectors; ++sector_id) {
            in.read(sector.data(), static_cast<std::streamsize>(sector.size()));
            if (!in.good()) {
                throw std::runtime_error("Short read while extracting DiskANN sectors.");
            }
            const std::uint64_t remaining = args.num_vectors - processed;
            const std::uint64_t nodes_in_sector =
                std::min<std::uint64_t>(layout.metadata().nodes_per_sector, remaining);
            for (std::uint64_t local = 0; local < nodes_in_sector; ++local) {
                neighbors.fill(topoanns::kInvalidNodeId);
                const char* node = sector.data() + local * layout.metadata().max_node_len;
                std::uint32_t degree = 0;
                std::memcpy(&degree, node + coord_bytes, sizeof(degree));
                const std::uint32_t* raw_neighbors =
                    reinterpret_cast<const std::uint32_t*>(node + coord_bytes + sizeof(std::uint32_t));
                const std::size_t limit =
                    std::min<std::size_t>(degree, topoanns::kFixedTopologyDegree);
                for (std::size_t i = 0; i < limit; ++i) {
                    if (raw_neighbors[i] < args.num_vectors) {
                        neighbors[i] = raw_neighbors[i];
                    } else {
                        ++filtered_neighbors;
                    }
                }
                out.write(reinterpret_cast<const char*>(neighbors.data()),
                          static_cast<std::streamsize>(sizeof(std::uint32_t) * neighbors.size()));
                ++processed;
            }
            if (processed == args.num_vectors ||
                processed / kProgressStepVectors !=
                    (processed - nodes_in_sector) / kProgressStepVectors) {
                std::cout << "[topoanns_build_float] extracted topology " << processed << " / "
                          << args.num_vectors << std::endl;
            }
        }
    } else {
        std::vector<char> node(layout.sectors_per_node() * topoanns::kDefaultPageSizeBytes, 0);
        for (std::uint64_t node_id = 0; node_id < args.num_vectors; ++node_id) {
            neighbors.fill(topoanns::kInvalidNodeId);
            in.seekg(static_cast<std::streamoff>(layout.NodeOffsetBytes(node_id)), std::ios::beg);
            in.read(node.data(), static_cast<std::streamsize>(node.size()));
            if (!in.good()) {
                throw std::runtime_error("Short read while extracting multi-sector DiskANN nodes.");
            }
            std::uint32_t degree = 0;
            std::memcpy(&degree, node.data() + coord_bytes, sizeof(degree));
            const std::uint32_t* raw_neighbors = reinterpret_cast<const std::uint32_t*>(
                node.data() + coord_bytes + sizeof(std::uint32_t));
            const std::size_t limit =
                std::min<std::size_t>(degree, topoanns::kFixedTopologyDegree);
            for (std::size_t i = 0; i < limit; ++i) {
                if (raw_neighbors[i] < args.num_vectors) {
                    neighbors[i] = raw_neighbors[i];
                } else {
                    ++filtered_neighbors;
                }
            }
            out.write(reinterpret_cast<const char*>(neighbors.data()),
                      static_cast<std::streamsize>(sizeof(std::uint32_t) * neighbors.size()));
            ++processed;
        }
    }

    std::cout << "[topoanns_build_float] filtered out-of-range neighbors: "
              << filtered_neighbors << std::endl;
}

void WriteManifest(const Args& args,
                   const FloatBinMetadata& metadata,
                   const topoanns::DiskannDiskIndexLayout& disk_layout,
                   const topoanns::VectorPageLayout& vector_layout) {
    const auto path = args.output_dir / "manifest.json";
    std::ofstream out(path, std::ios::trunc);
    out << "{\n"
        << "  \"num_vectors\": " << args.num_vectors << ",\n"
        << "  \"dim\": " << metadata.dim << ",\n"
        << "  \"vector_scalar\": \"float32\",\n"
        << "  \"topology_degree\": " << topoanns::kFixedTopologyDegree << ",\n"
        << "  \"vectors_per_page\": " << vector_layout.vectors_per_page() << ",\n"
        << "  \"source_base_fbin\": \"" << args.base_fbin.string() << "\",\n"
        << "  \"source_disk_index\": \"" << args.disk_index.string() << "\",\n"
        << "  \"diskann_medoid\": " << disk_layout.metadata().medoid_id << ",\n"
        << "  \"diskann_neighbor_capacity\": " << disk_layout.neighbor_capacity() << "\n"
        << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Args args = ParseArgs(argc, argv);
        std::filesystem::create_directories(args.output_dir);

        const FloatBinMetadata metadata = ReadFloatBinMetadata(args.base_fbin);
        const topoanns::DiskannDiskIndexLayout disk_layout =
            topoanns::DiskannDiskIndexLayout::Load(args.disk_index);
        if (metadata.dim != disk_layout.metadata().vector_dim) {
            throw std::runtime_error("float bin dim and DiskANN vector dim do not match.");
        }
        if (args.num_vectors > metadata.num_vectors ||
            args.num_vectors > disk_layout.metadata().num_nodes) {
            throw std::runtime_error("Requested num_vectors exceeds source dataset size.");
        }

        const topoanns::VectorPageLayout vector_layout =
            topoanns::VectorPageLayout::Create(metadata.dim, topoanns::ScalarKind::kFloat32);
        const auto vector_store_path = args.output_dir / "vectors.ssd";
        const auto topology_path = args.output_dir / "topology.bin";
        const auto pq_pivots_path = args.output_dir / "_pq_pivots.bin";
        const auto pq_compressed_path = args.output_dir / "_pq_compressed.bin";

        ConvertVectors(args, metadata, vector_store_path);
        ExtractTopology(args, disk_layout, topology_path);
        CopyFile(args.pq_pivots, pq_pivots_path);
        SlicePqCompressed(args.pq_compressed, pq_compressed_path, args.num_vectors);
        WriteManifest(args, metadata, disk_layout, vector_layout);

        std::cout << "[topoanns_build_float] done" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_build_float] " << e.what() << std::endl;
        return 1;
    }
}
