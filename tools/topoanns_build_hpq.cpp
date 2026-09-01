#include <immintrin.h>

#include "math_utils.h"
#include "pq.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kPqMetadataSizeBytes = 4096;
constexpr std::size_t kNumCenters = 256;
constexpr std::size_t kEncodeShardRows = 16384;
constexpr std::uint32_t kKmeansIterations = 15;

enum class DataFormat { kFbin, kBvecs };

struct Args {
    std::filesystem::path data;
    DataFormat data_format = DataFormat::kFbin;
    std::filesystem::path base_pivots;
    std::filesystem::path base_codes;
    std::filesystem::path outlier_pivots;
    std::filesystem::path hybrid_codes;
    std::filesystem::path selector_bits;
    std::size_t num_chunks = 0;
    std::size_t train_points = 1280000;
    std::size_t block_points = 65536;
    double outlier_fraction = 0.20;
    bool train_base_pq = false;
    bool resume = false;
};

struct MatrixMetadata {
    std::size_t rows = 0;
    std::size_t cols = 0;
};

template <typename T>
struct BinBlock {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> data;
};

struct PqPivots {
    std::size_t dim = 0;
    std::size_t chunks = 0;
    std::vector<float> tables;
    std::vector<float> centroid;
    std::vector<std::uint32_t> chunk_offsets;
};

[[noreturn]] void Usage() {
    std::cerr << "Usage: topoanns_build_hpq"
              << " --data <path> --data-format <fbin|bvecs>"
              << " --base-pivots <path> --base-codes <path>"
              << " --outlier-pivots <path> --hybrid-codes <path>"
              << " --selector-bits <path> --num-chunks <count>"
              << " [--train-points <count>] [--block-points <count>]"
              << " [--outlier-fraction <value>] [--train-base-pq] [--resume]"
              << std::endl;
    std::exit(1);
}

Args ParseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        const auto value = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("Missing value for ") + name);
            }
            return argv[++i];
        };
        if (flag == "--data") args.data = value("--data");
        else if (flag == "--data-format") {
            const std::string format = value("--data-format");
            if (format == "fbin") args.data_format = DataFormat::kFbin;
            else if (format == "bvecs") args.data_format = DataFormat::kBvecs;
            else throw std::runtime_error("Unsupported data format: " + format);
        } else if (flag == "--base-pivots") args.base_pivots = value("--base-pivots");
        else if (flag == "--base-codes") args.base_codes = value("--base-codes");
        else if (flag == "--outlier-pivots") args.outlier_pivots = value("--outlier-pivots");
        else if (flag == "--hybrid-codes") args.hybrid_codes = value("--hybrid-codes");
        else if (flag == "--selector-bits") args.selector_bits = value("--selector-bits");
        else if (flag == "--num-chunks") args.num_chunks = std::stoull(value("--num-chunks"));
        else if (flag == "--train-points") args.train_points = std::stoull(value("--train-points"));
        else if (flag == "--block-points") args.block_points = std::stoull(value("--block-points"));
        else if (flag == "--outlier-fraction") args.outlier_fraction = std::stod(value("--outlier-fraction"));
        else if (flag == "--train-base-pq") args.train_base_pq = true;
        else if (flag == "--resume") args.resume = true;
        else throw std::runtime_error("Unknown flag: " + std::string(flag));
    }
    if (args.data.empty() || args.base_pivots.empty() || args.base_codes.empty() ||
        args.outlier_pivots.empty() || args.hybrid_codes.empty() ||
        args.selector_bits.empty() || args.num_chunks == 0 || args.block_points == 0 ||
        !(args.outlier_fraction > 0.0 && args.outlier_fraction < 1.0)) {
        Usage();
    }
    return args;
}

std::pair<std::size_t, std::size_t> ReadBinMetadata(const std::filesystem::path& path,
                                                    std::size_t offset = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open " + path.string());
    in.seekg(static_cast<std::streamoff>(offset));
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!in.good() || rows < 0 || cols < 0) {
        throw std::runtime_error("Invalid metadata in " + path.string());
    }
    return {static_cast<std::size_t>(rows), static_cast<std::size_t>(cols)};
}

template <typename T>
BinBlock<T> ReadBinBlock(const std::filesystem::path& path, std::size_t offset = 0) {
    const auto [rows, cols] = ReadBinMetadata(path, offset);
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(offset + sizeof(std::int32_t) * 2));
    BinBlock<T> block{rows, cols, std::vector<T>(rows * cols)};
    in.read(reinterpret_cast<char*>(block.data.data()),
            static_cast<std::streamsize>(block.data.size() * sizeof(T)));
    if (!in.good()) throw std::runtime_error("Short read from " + path.string());
    return block;
}

MatrixMetadata ReadDataMetadata(const Args& args) {
    if (args.data_format == DataFormat::kFbin) {
        const auto [rows, cols] = ReadBinMetadata(args.data);
        return {rows, cols};
    }
    std::ifstream in(args.data, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Failed to open " + args.data.string());
    const std::size_t bytes = static_cast<std::size_t>(in.tellg());
    in.seekg(0);
    std::int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in.good() || dim <= 0 || bytes % (sizeof(dim) + static_cast<std::size_t>(dim)) != 0) {
        throw std::runtime_error("Invalid bvecs file " + args.data.string());
    }
    return {bytes / (sizeof(dim) + static_cast<std::size_t>(dim)),
            static_cast<std::size_t>(dim)};
}

void ReadDataRows(const Args& args, const MatrixMetadata& metadata, std::size_t row_begin,
                  std::size_t row_count, std::vector<float>* output) {
    output->assign(row_count * metadata.cols, 0.0f);
    std::ifstream in(args.data, std::ios::binary);
    if (!in) throw std::runtime_error("Failed to open " + args.data.string());
    if (args.data_format == DataFormat::kFbin) {
        const std::size_t offset = sizeof(std::int32_t) * 2 +
                                   row_begin * metadata.cols * sizeof(float);
        in.seekg(static_cast<std::streamoff>(offset));
        in.read(reinterpret_cast<char*>(output->data()),
                static_cast<std::streamsize>(output->size() * sizeof(float)));
    } else {
        const std::size_t row_bytes = sizeof(std::int32_t) + metadata.cols;
        in.seekg(static_cast<std::streamoff>(row_begin * row_bytes));
        std::vector<std::uint8_t> row(metadata.cols);
        for (std::size_t i = 0; i < row_count; ++i) {
            std::int32_t dim = 0;
            in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
            in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size()));
            if (dim != static_cast<std::int32_t>(metadata.cols)) {
                throw std::runtime_error("Inconsistent bvecs dimension.");
            }
            std::copy(row.begin(), row.end(), output->begin() + i * metadata.cols);
        }
    }
    if (!in.good()) throw std::runtime_error("Short data read from " + args.data.string());
}

void ReadCodeRows(const std::filesystem::path& path, std::size_t row_begin,
                  std::size_t row_count, std::size_t chunks,
                  std::vector<std::uint8_t>* output) {
    output->resize(row_count * chunks);
    std::ifstream in(path, std::ios::binary);
    in.seekg(static_cast<std::streamoff>(sizeof(std::int32_t) * 2 + row_begin * chunks));
    in.read(reinterpret_cast<char*>(output->data()),
            static_cast<std::streamsize>(output->size()));
    if (!in.good()) throw std::runtime_error("Short base-code read.");
}

PqPivots LoadPivots(const std::filesystem::path& path) {
    const auto [centers, dim] = ReadBinMetadata(path, kPqMetadataSizeBytes);
    if (centers != kNumCenters) throw std::runtime_error("Unexpected PQ center count.");
    const BinBlock<std::size_t> offsets = ReadBinBlock<std::size_t>(path);
    const bool old_format = offsets.rows == 5;
    if (!old_format && offsets.rows != 4) throw std::runtime_error("Invalid PQ offsets.");
    const BinBlock<float> tables = ReadBinBlock<float>(path, offsets.data[0]);
    const BinBlock<float> centroid = ReadBinBlock<float>(path, offsets.data[1]);
    const BinBlock<std::uint32_t> chunks =
        ReadBinBlock<std::uint32_t>(path, offsets.data[old_format ? 3 : 2]);
    return {dim, chunks.rows - 1, tables.data, centroid.data, chunks.data};
}

void WritePivots(const std::filesystem::path& path, const PqPivots& pivots) {
    std::vector<std::size_t> offsets(4, 0);
    offsets[0] = kPqMetadataSizeBytes;
    offsets[1] = offsets[0] + diskann::save_bin<float>(
        path.string(), const_cast<float*>(pivots.tables.data()), kNumCenters, pivots.dim,
        offsets[0]);
    offsets[2] = offsets[1] + diskann::save_bin<float>(
        path.string(), const_cast<float*>(pivots.centroid.data()), pivots.dim, 1, offsets[1]);
    offsets[3] = offsets[2] + diskann::save_bin<std::uint32_t>(
        path.string(), const_cast<std::uint32_t*>(pivots.chunk_offsets.data()),
        pivots.chunk_offsets.size(), 1, offsets[2]);
    diskann::save_bin<std::size_t>(path.string(), offsets.data(), offsets.size(), 1, 0);
}

PqPivots TrainOutlierPivots(const PqPivots& base, const std::vector<float>& vectors,
                            const std::vector<std::uint8_t>& base_codes,
                            std::size_t rows, double outlier_fraction) {
    PqPivots outlier = base;
    std::vector<float> errors(rows);
    std::vector<std::size_t> indices(rows);
    for (std::size_t chunk = 0; chunk < base.chunks; ++chunk) {
        const std::size_t begin = base.chunk_offsets[chunk];
        const std::size_t end = base.chunk_offsets[chunk + 1];
        const std::size_t chunk_dim = end - begin;
        for (std::size_t row = 0; row < rows; ++row) {
            float error = 0.0f;
            const std::size_t code = base_codes[row * base.chunks + chunk];
            for (std::size_t dim = begin; dim < end; ++dim) {
                const float value = vectors[row * base.dim + dim] - base.centroid[dim];
                const float diff = value - base.tables[code * base.dim + dim];
                error += diff * diff;
            }
            errors[row] = error;
            indices[row] = row;
        }
        const std::size_t selected = std::min<std::size_t>(
            rows, std::min<std::size_t>(
                      256000, std::max<std::size_t>(
                                  kNumCenters, static_cast<std::size_t>(
                                                   std::floor(rows * outlier_fraction)))));
        if (selected < rows) {
            std::nth_element(indices.begin(), indices.begin() + selected, indices.end(),
                             [&](std::size_t lhs, std::size_t rhs) {
                                 return errors[lhs] > errors[rhs];
                             });
        }
        std::vector<float> training(selected * chunk_dim);
        for (std::size_t i = 0; i < selected; ++i) {
            const std::size_t row = indices[i];
            for (std::size_t d = 0; d < chunk_dim; ++d) {
                training[i * chunk_dim + d] =
                    vectors[row * base.dim + begin + d] - base.centroid[begin + d];
            }
        }
        std::vector<float> centers(kNumCenters * chunk_dim);
        kmeans::selecting_pivots(training.data(), selected, chunk_dim,
                                centers.data(), kNumCenters);
        kmeans::run_lloyds(training.data(), selected, chunk_dim, centers.data(),
                           kNumCenters, kKmeansIterations, nullptr, nullptr);
        for (std::size_t center = 0; center < kNumCenters; ++center) {
            std::memcpy(outlier.tables.data() + center * base.dim + begin,
                        centers.data() + center * chunk_dim, chunk_dim * sizeof(float));
        }
        std::cout << "[topoanns_build_hpq] trained_chunk=" << chunk + 1 << "/"
                  << base.chunks << " selected=" << selected << std::endl;
    }
    return outlier;
}

std::vector<std::uint8_t> EncodeCodes(const PqPivots& pivots,
                                      const std::vector<float>& vectors,
                                      std::size_t rows) {
    std::vector<std::uint8_t> codes(rows * pivots.chunks);
    const std::size_t shards = (rows + kEncodeShardRows - 1) / kEncodeShardRows;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::int64_t task_i = 0;
         task_i < static_cast<std::int64_t>(pivots.chunks * shards); ++task_i) {
        const std::size_t task = static_cast<std::size_t>(task_i);
        const std::size_t chunk = task / shards;
        const std::size_t shard = task % shards;
        const std::size_t row_begin = shard * kEncodeShardRows;
        const std::size_t shard_rows = std::min(kEncodeShardRows, rows - row_begin);
        const std::size_t begin = pivots.chunk_offsets[chunk];
        const std::size_t end = pivots.chunk_offsets[chunk + 1];
        const std::size_t chunk_dim = end - begin;
        std::vector<float> chunk_data(shard_rows * chunk_dim);
        std::vector<float> centers(kNumCenters * chunk_dim);
        for (std::size_t row = 0; row < shard_rows; ++row) {
            for (std::size_t d = 0; d < chunk_dim; ++d) {
                chunk_data[row * chunk_dim + d] =
                    vectors[(row_begin + row) * pivots.dim + begin + d] -
                    pivots.centroid[begin + d];
            }
        }
        for (std::size_t center = 0; center < kNumCenters; ++center) {
            std::memcpy(centers.data() + center * chunk_dim,
                        pivots.tables.data() + center * pivots.dim + begin,
                        chunk_dim * sizeof(float));
        }
        std::vector<std::uint32_t> closest(shard_rows);
        math_utils::compute_closest_centers(chunk_data.data(), shard_rows, chunk_dim,
                                            centers.data(), kNumCenters, 1,
                                            closest.data(), nullptr, nullptr);
        for (std::size_t row = 0; row < shard_rows; ++row) {
            codes[(row_begin + row) * pivots.chunks + chunk] =
                static_cast<std::uint8_t>(closest[row]);
        }
    }
    return codes;
}

std::size_t CompletedRows(const std::filesystem::path& path, std::size_t row_bytes) {
    if (!std::filesystem::exists(path)) return 0;
    const std::uintmax_t bytes = std::filesystem::file_size(path);
    if (bytes < sizeof(std::int32_t) * 2 || (bytes - sizeof(std::int32_t) * 2) % row_bytes != 0) {
        throw std::runtime_error("Invalid partial output size: " + path.string());
    }
    return static_cast<std::size_t>((bytes - sizeof(std::int32_t) * 2) / row_bytes);
}

void WriteHeader(std::fstream& out, std::int32_t rows, std::int32_t cols) {
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
}

int Main(const Args& args) {
    const MatrixMetadata metadata = ReadDataMetadata(args);
    const std::size_t train_rows = std::min(args.train_points, metadata.rows);
    std::vector<float> training;
    if (args.train_base_pq && !std::filesystem::exists(args.base_pivots)) {
        ReadDataRows(args, metadata, 0, train_rows, &training);
        const int rc = diskann::generate_pq_pivots(
            training.data(), train_rows, static_cast<unsigned>(metadata.cols),
            static_cast<unsigned>(kNumCenters), static_cast<unsigned>(args.num_chunks),
            kKmeansIterations, args.base_pivots.string(), true);
        if (rc != 0) throw std::runtime_error("Failed to train base PQ pivots.");
    }
    const PqPivots base = LoadPivots(args.base_pivots);
    if (metadata.cols != base.dim || base.chunks != args.num_chunks) {
        throw std::runtime_error("Dataset and base-PQ shapes do not match.");
    }
    if (std::filesystem::exists(args.base_codes)) {
        const auto [code_rows, code_cols] = ReadBinMetadata(args.base_codes);
        if (metadata.rows != code_rows || code_cols != args.num_chunks) {
            throw std::runtime_error("Dataset and base-PQ code shapes do not match.");
        }
    } else if (!args.train_base_pq) {
        throw std::runtime_error("Base PQ codes do not exist.");
    }

    PqPivots outlier;
    if (args.resume && std::filesystem::exists(args.outlier_pivots)) {
        outlier = LoadPivots(args.outlier_pivots);
    } else {
        std::vector<std::uint8_t> training_codes;
        if (training.empty()) ReadDataRows(args, metadata, 0, train_rows, &training);
        if (args.train_base_pq) {
            training_codes = EncodeCodes(base, training, train_rows);
        } else {
            ReadCodeRows(args.base_codes, 0, train_rows, base.chunks, &training_codes);
        }
        outlier = TrainOutlierPivots(base, training, training_codes, train_rows,
                                     args.outlier_fraction);
        WritePivots(args.outlier_pivots, outlier);
    }

    const std::size_t selector_stride = (base.chunks + 7U) / 8U;
    std::size_t row_begin = 0;
    if (args.resume) {
        const std::size_t code_done = CompletedRows(args.hybrid_codes, base.chunks);
        const std::size_t selector_done = CompletedRows(args.selector_bits, selector_stride);
        if (code_done != selector_done) throw std::runtime_error("HPQ partial outputs differ.");
        if (args.train_base_pq) {
            const std::size_t base_done = CompletedRows(args.base_codes, base.chunks);
            if (base_done != code_done) {
                throw std::runtime_error("Base-PQ and HPQ partial outputs differ.");
            }
        }
        row_begin = code_done;
    }
    if (row_begin > metadata.rows) throw std::runtime_error("HPQ output exceeds dataset.");
    if (row_begin == metadata.rows) return 0;

    std::fstream codes;
    std::fstream selectors;
    std::fstream base_codes_out;
    const auto mode = row_begin == 0
                          ? (std::ios::binary | std::ios::out | std::ios::trunc)
                          : (std::ios::binary | std::ios::in | std::ios::out);
    codes.open(args.hybrid_codes, mode);
    selectors.open(args.selector_bits, mode);
    if (args.train_base_pq) base_codes_out.open(args.base_codes, mode);
    if (!codes || !selectors || (args.train_base_pq && !base_codes_out)) {
        throw std::runtime_error("Failed to open PQ outputs.");
    }
    if (row_begin == 0) {
        WriteHeader(codes, static_cast<std::int32_t>(metadata.rows),
                    static_cast<std::int32_t>(base.chunks));
        if (args.train_base_pq) {
            WriteHeader(base_codes_out, static_cast<std::int32_t>(metadata.rows),
                        static_cast<std::int32_t>(base.chunks));
        }
        WriteHeader(selectors, static_cast<std::int32_t>(metadata.rows),
                    static_cast<std::int32_t>(selector_stride));
    } else {
        codes.seekp(static_cast<std::streamoff>(sizeof(std::int32_t) * 2 + row_begin * base.chunks));
        if (args.train_base_pq) {
            base_codes_out.seekp(static_cast<std::streamoff>(
                sizeof(std::int32_t) * 2 + row_begin * base.chunks));
        }
        selectors.seekp(static_cast<std::streamoff>(sizeof(std::int32_t) * 2 +
                                                    row_begin * selector_stride));
    }

    std::uint64_t selected_subspaces = 0;
    std::vector<float> vectors;
    std::vector<std::uint8_t> base_block_codes;
    for (; row_begin < metadata.rows; row_begin += args.block_points) {
        const std::size_t rows = std::min(args.block_points, metadata.rows - row_begin);
        ReadDataRows(args, metadata, row_begin, rows, &vectors);
        if (args.train_base_pq) {
            base_block_codes = EncodeCodes(base, vectors, rows);
        } else {
            ReadCodeRows(args.base_codes, row_begin, rows, base.chunks, &base_block_codes);
        }
        std::vector<std::uint8_t> hybrid = base_block_codes;
        std::vector<std::uint8_t> bits(rows * selector_stride, 0);
        std::vector<std::uint8_t> choices(rows * base.chunks, 0);
        const std::size_t shards = (rows + kEncodeShardRows - 1) / kEncodeShardRows;

#pragma omp parallel for schedule(dynamic, 1) reduction(+ : selected_subspaces)
        for (std::int64_t task_i = 0;
             task_i < static_cast<std::int64_t>(base.chunks * shards); ++task_i) {
            const std::size_t task = static_cast<std::size_t>(task_i);
            const std::size_t chunk = task / shards;
            const std::size_t shard = task % shards;
            const std::size_t row_begin_in_block = shard * kEncodeShardRows;
            const std::size_t shard_rows =
                std::min(kEncodeShardRows, rows - row_begin_in_block);
            const std::size_t begin = base.chunk_offsets[chunk];
            const std::size_t end = base.chunk_offsets[chunk + 1];
            const std::size_t chunk_dim = end - begin;
            std::vector<float> chunk_data(shard_rows * chunk_dim);
            std::vector<float> centers(kNumCenters * chunk_dim);
            for (std::size_t row = 0; row < shard_rows; ++row) {
                for (std::size_t d = 0; d < chunk_dim; ++d) {
                    chunk_data[row * chunk_dim + d] =
                        vectors[(row_begin_in_block + row) * base.dim + begin + d] -
                        base.centroid[begin + d];
                }
            }
            for (std::size_t center = 0; center < kNumCenters; ++center) {
                std::memcpy(centers.data() + center * chunk_dim,
                            outlier.tables.data() + center * base.dim + begin,
                            chunk_dim * sizeof(float));
            }
            std::vector<std::uint32_t> closest(shard_rows);
            math_utils::compute_closest_centers(chunk_data.data(), shard_rows, chunk_dim,
                                                centers.data(), kNumCenters, 1,
                                                closest.data(), nullptr, nullptr);
            for (std::size_t row = 0; row < shard_rows; ++row) {
                const std::size_t output_row = row_begin_in_block + row;
                const std::size_t base_code =
                    base_block_codes[output_row * base.chunks + chunk];
                float base_error = 0.0f;
                float outlier_error = 0.0f;
                for (std::size_t d = 0; d < chunk_dim; ++d) {
                    const float value = chunk_data[row * chunk_dim + d];
                    const float base_diff =
                        value - base.tables[base_code * base.dim + begin + d];
                    const float outlier_diff =
                        value - centers[static_cast<std::size_t>(closest[row]) * chunk_dim + d];
                    base_error += base_diff * base_diff;
                    outlier_error += outlier_diff * outlier_diff;
                }
                if (outlier_error < base_error) {
                    hybrid[output_row * base.chunks + chunk] =
                        static_cast<std::uint8_t>(closest[row]);
                    choices[output_row * base.chunks + chunk] = 1;
                    ++selected_subspaces;
                }
            }
        }
        for (std::size_t row = 0; row < rows; ++row) {
            for (std::size_t chunk = 0; chunk < base.chunks; ++chunk) {
                if (choices[row * base.chunks + chunk] != 0) {
                    bits[row * selector_stride + (chunk >> 3U)] |=
                        static_cast<std::uint8_t>(1U << (chunk & 7U));
                }
            }
        }
        if (args.train_base_pq) {
            base_codes_out.write(reinterpret_cast<const char*>(base_block_codes.data()),
                                 static_cast<std::streamsize>(base_block_codes.size()));
        }
        codes.write(reinterpret_cast<const char*>(hybrid.data()),
                    static_cast<std::streamsize>(hybrid.size()));
        selectors.write(reinterpret_cast<const char*>(bits.data()),
                        static_cast<std::streamsize>(bits.size()));
        std::cout << "[topoanns_build_hpq] encoded_rows=" << row_begin + rows << "/"
                  << metadata.rows << std::endl;
    }
    std::cout << "[topoanns_build_hpq] selected_subspaces=" << selected_subspaces << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(ParseArgs(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "[topoanns_build_hpq] " << error.what() << std::endl;
        return 1;
    }
}
