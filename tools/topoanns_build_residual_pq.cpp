#include <immintrin.h>

#include "math_utils.h"
#include "pq.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
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
constexpr std::size_t kNumPqCentroids = 256;
constexpr std::uint32_t kResidualPqKmeansIters = 15;

struct Args {
    std::filesystem::path data_bin;
    std::filesystem::path base_pq_pivots;
    std::filesystem::path base_pq_codes;
    std::filesystem::path residual_pq_pivots;
    std::filesystem::path residual_pq_codes;
    std::filesystem::path residual_pq_error;
    std::uint32_t num_chunks = 0;
    std::size_t train_points = 200000;
    std::size_t block_points = 65536;
    bool resume = false;
};

template <typename T>
struct BinBlock {
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::vector<T> data;
};

struct PqPivotData {
    std::size_t ndims = 0;
    std::size_t num_chunks = 0;
    std::vector<float> tables_row_major;
    std::vector<std::uint32_t> chunk_offsets;
    std::vector<float> centroid;
};

struct FbinMetadata {
    std::size_t rows = 0;
    std::size_t cols = 0;
};

struct OutputFileState {
    bool exists = false;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t encoded_rows = 0;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_build_residual_pq"
        << " --data-bin <path>"
        << " --base-pq-pivots <path>"
        << " --base-pq-codes <path>"
        << " --residual-pq-pivots <path>"
        << " --residual-pq-codes <path>"
        << " --residual-pq-error <path>"
        << " --num-chunks <count>"
        << " [--train-points <count>]"
        << " [--block-points <count>]"
        << " [--resume]"
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
        if (flag == "--data-bin") {
            args.data_bin = read_value("--data-bin");
        } else if (flag == "--base-pq-pivots") {
            args.base_pq_pivots = read_value("--base-pq-pivots");
        } else if (flag == "--base-pq-codes") {
            args.base_pq_codes = read_value("--base-pq-codes");
        } else if (flag == "--residual-pq-pivots") {
            args.residual_pq_pivots = read_value("--residual-pq-pivots");
        } else if (flag == "--residual-pq-codes") {
            args.residual_pq_codes = read_value("--residual-pq-codes");
        } else if (flag == "--residual-pq-error") {
            args.residual_pq_error = read_value("--residual-pq-error");
        } else if (flag == "--num-chunks") {
            args.num_chunks = static_cast<std::uint32_t>(std::stoul(read_value("--num-chunks")));
        } else if (flag == "--train-points") {
            args.train_points = std::stoull(read_value("--train-points"));
        } else if (flag == "--block-points") {
            args.block_points = std::stoull(read_value("--block-points"));
        } else if (flag == "--resume") {
            args.resume = true;
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }

    if (args.data_bin.empty() || args.base_pq_pivots.empty() || args.base_pq_codes.empty() ||
        args.residual_pq_pivots.empty() || args.residual_pq_codes.empty() ||
        args.residual_pq_error.empty() || args.num_chunks == 0 || args.block_points == 0) {
        Usage();
    }
    return args;
}

template <typename T>
BinBlock<T> ReadBinBlock(const std::filesystem::path& path, std::size_t offset = 0) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }

    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::int32_t rows_i32 = 0;
    std::int32_t cols_i32 = 0;
    in.read(reinterpret_cast<char*>(&rows_i32), sizeof(rows_i32));
    in.read(reinterpret_cast<char*>(&cols_i32), sizeof(cols_i32));
    if (!in.good() || rows_i32 < 0 || cols_i32 < 0) {
        throw std::runtime_error("Invalid metadata in " + path.string());
    }

    BinBlock<T> block;
    block.rows = static_cast<std::size_t>(rows_i32);
    block.cols = static_cast<std::size_t>(cols_i32);
    block.data.resize(block.rows * block.cols);
    if (!block.data.empty()) {
        in.read(reinterpret_cast<char*>(block.data.data()),
                static_cast<std::streamsize>(block.data.size() * sizeof(T)));
        if (!in.good()) {
            throw std::runtime_error("Short read in " + path.string());
        }
    }
    return block;
}

std::pair<std::size_t, std::size_t> ReadBinMetadata(const std::filesystem::path& path,
                                                    std::size_t offset) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    std::int32_t rows_i32 = 0;
    std::int32_t cols_i32 = 0;
    in.read(reinterpret_cast<char*>(&rows_i32), sizeof(rows_i32));
    in.read(reinterpret_cast<char*>(&cols_i32), sizeof(cols_i32));
    if (!in.good() || rows_i32 < 0 || cols_i32 < 0) {
        throw std::runtime_error("Invalid metadata in " + path.string());
    }
    return {static_cast<std::size_t>(rows_i32), static_cast<std::size_t>(cols_i32)};
}

OutputFileState InspectOutputFile(const std::filesystem::path& path, std::size_t element_bytes) {
    OutputFileState state;
    if (!std::filesystem::exists(path)) {
        return state;
    }
    state.exists = true;
    const auto [rows, cols] = ReadBinMetadata(path, 0);
    state.rows = rows;
    state.cols = cols;
    const std::uintmax_t size = std::filesystem::file_size(path);
    const std::size_t header_bytes = sizeof(std::int32_t) * 2;
    if (size < header_bytes) {
        throw std::runtime_error("Output file is smaller than header: " + path.string());
    }
    const std::uintmax_t payload_bytes = size - header_bytes;
    const std::uintmax_t row_bytes = static_cast<std::uintmax_t>(cols) * element_bytes;
    if (row_bytes == 0 || (payload_bytes % row_bytes) != 0) {
        throw std::runtime_error("Output file has invalid payload size: " + path.string());
    }
    state.encoded_rows = static_cast<std::size_t>(payload_bytes / row_bytes);
    return state;
}

PqPivotData LoadPqPivots(const std::filesystem::path& path) {
    const auto [num_centroids, dim] = ReadBinMetadata(path, kPqMetadataSizeBytes);
    if (num_centroids != kNumPqCentroids) {
        throw std::runtime_error("Unexpected PQ centroid count in " + path.string());
    }

    const BinBlock<std::size_t> offsets = ReadBinBlock<std::size_t>(path);
    if (offsets.rows != 4 && offsets.rows != 5) {
        throw std::runtime_error("Unexpected pivot offset count in " + path.string());
    }
    const bool use_old_filetype = offsets.rows == 5;
    const std::size_t tables_offset = offsets.data[0];
    const std::size_t centroid_offset = offsets.data[1];
    const std::size_t chunk_offsets_offset = offsets.data[use_old_filetype ? 3 : 2];

    const BinBlock<float> tables = ReadBinBlock<float>(path, tables_offset);
    const BinBlock<float> centroid = ReadBinBlock<float>(path, centroid_offset);
    const BinBlock<std::uint32_t> chunk_offsets =
        ReadBinBlock<std::uint32_t>(path, chunk_offsets_offset);

    if (tables.rows != kNumPqCentroids || tables.cols != dim) {
        throw std::runtime_error("Unexpected PQ table shape in " + path.string());
    }
    if (centroid.rows != dim || centroid.cols != 1) {
        throw std::runtime_error("Unexpected PQ centroid shape in " + path.string());
    }
    if (chunk_offsets.cols != 1 || chunk_offsets.rows < 2) {
        throw std::runtime_error("Unexpected PQ chunk-offset shape in " + path.string());
    }

    PqPivotData result;
    result.ndims = dim;
    result.num_chunks = chunk_offsets.rows - 1;
    result.tables_row_major = tables.data;
    result.chunk_offsets = chunk_offsets.data;
    result.centroid = centroid.data;
    return result;
}

FbinMetadata ReadFbinMetadata(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    std::int32_t rows_i32 = 0;
    std::int32_t cols_i32 = 0;
    in.read(reinterpret_cast<char*>(&rows_i32), sizeof(rows_i32));
    in.read(reinterpret_cast<char*>(&cols_i32), sizeof(cols_i32));
    if (!in.good() || rows_i32 < 0 || cols_i32 < 0) {
        throw std::runtime_error("Invalid fbin metadata in " + path.string());
    }
    return {static_cast<std::size_t>(rows_i32), static_cast<std::size_t>(cols_i32)};
}

void ReadFloatRows(const std::filesystem::path& path,
                   std::size_t row_begin,
                   std::size_t row_count,
                   std::size_t dim,
                   std::vector<float>* out) {
    out->assign(row_count * dim, 0.0f);
    if (row_count == 0) {
        return;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    const std::size_t offset = sizeof(std::int32_t) * 2 + row_begin * dim * sizeof(float);
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    in.read(reinterpret_cast<char*>(out->data()),
            static_cast<std::streamsize>(out->size() * sizeof(float)));
    if (!in.good()) {
        throw std::runtime_error("Short read from " + path.string());
    }
}

void ReadCodeRows(const std::filesystem::path& path,
                  std::size_t row_begin,
                  std::size_t row_count,
                  std::size_t num_chunks,
                  std::vector<std::uint8_t>* out) {
    out->assign(row_count * num_chunks, 0U);
    if (row_count == 0) {
        return;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    const std::size_t offset = sizeof(std::int32_t) * 2 + row_begin * num_chunks;
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    in.read(reinterpret_cast<char*>(out->data()),
            static_cast<std::streamsize>(out->size()));
    if (!in.good()) {
        throw std::runtime_error("Short PQ-code read from " + path.string());
    }
}

void ReconstructVectors(const PqPivotData& pivots,
                        const std::vector<std::uint8_t>& codes,
                        std::size_t row_count,
                        std::vector<float>* out) {
    out->assign(row_count * pivots.ndims, 0.0f);
    for (std::size_t row = 0; row < row_count; ++row) {
        float* dst = out->data() + row * pivots.ndims;
        const std::uint8_t* code_ptr = codes.data() + row * pivots.num_chunks;
        for (std::size_t chunk = 0; chunk < pivots.num_chunks; ++chunk) {
            const std::size_t begin = pivots.chunk_offsets[chunk];
            const std::size_t end = pivots.chunk_offsets[chunk + 1];
            const std::size_t center = code_ptr[chunk];
            for (std::size_t dim = begin; dim < end; ++dim) {
                dst[dim] = pivots.centroid[dim] +
                           pivots.tables_row_major[center * pivots.ndims + dim];
            }
        }
    }
}

std::vector<float> ComputeResiduals(const std::vector<float>& base,
                                    const std::vector<float>& reconstructed) {
    if (base.size() != reconstructed.size()) {
        throw std::runtime_error("Residual size mismatch.");
    }
    std::vector<float> residuals(base.size(), 0.0f);
    for (std::size_t i = 0; i < base.size(); ++i) {
        residuals[i] = base[i] - reconstructed[i];
    }
    return residuals;
}

bool HasUniformChunkLayout(const PqPivotData& pivots) {
    if (pivots.chunk_offsets.size() < 2) {
        return false;
    }
    const std::size_t chunk_size = pivots.chunk_offsets[1] - pivots.chunk_offsets[0];
    for (std::size_t chunk = 1; chunk + 1 < pivots.chunk_offsets.size(); ++chunk) {
        if (pivots.chunk_offsets[chunk + 1] - pivots.chunk_offsets[chunk] != chunk_size) {
            return false;
        }
    }
    return true;
}

void EncodeResidualBlock(const PqPivotData& residual_pq,
                         const std::vector<float>& block_residuals,
                         std::size_t row_count,
                         std::vector<std::uint8_t>* out_codes) {
    out_codes->assign(row_count * residual_pq.num_chunks, 0U);
    if (row_count == 0) {
        return;
    }

    if (HasUniformChunkLayout(residual_pq)) {
        const int encode_rc = diskann::generate_pq_data_from_pivots_simplified(
            block_residuals.data(), row_count, residual_pq.tables_row_major.data(),
            kNumPqCentroids * residual_pq.ndims, residual_pq.ndims, residual_pq.num_chunks,
            *out_codes);
        if (encode_rc != 0) {
            throw std::runtime_error("generate_pq_data_from_pivots_simplified returned non-zero.");
        }
        return;
    }

    std::vector<float> chunk_pivots;
    std::vector<float> chunk_data;
    std::vector<std::uint32_t> closest_centers(row_count, 0U);
    for (std::size_t chunk = 0; chunk < residual_pq.num_chunks; ++chunk) {
        const std::size_t dim_begin = residual_pq.chunk_offsets[chunk];
        const std::size_t dim_end = residual_pq.chunk_offsets[chunk + 1];
        const std::size_t chunk_dim = dim_end - dim_begin;
        chunk_pivots.assign(kNumPqCentroids * chunk_dim, 0.0f);
        chunk_data.assign(row_count * chunk_dim, 0.0f);

        for (std::size_t center = 0; center < kNumPqCentroids; ++center) {
            float* pivot_ptr = chunk_pivots.data() + center * chunk_dim;
            for (std::size_t d = 0; d < chunk_dim; ++d) {
                const std::size_t dim = dim_begin + d;
                pivot_ptr[d] = residual_pq.tables_row_major[center * residual_pq.ndims + dim] +
                               residual_pq.centroid[dim];
            }
        }
        for (std::size_t row = 0; row < row_count; ++row) {
            const float* residual_ptr = block_residuals.data() + row * residual_pq.ndims + dim_begin;
            std::memcpy(chunk_data.data() + row * chunk_dim, residual_ptr,
                        chunk_dim * sizeof(float));
        }

        math_utils::compute_closest_centers(chunk_data.data(), row_count, chunk_dim,
                                            chunk_pivots.data(), kNumPqCentroids, 1,
                                            closest_centers.data());
        for (std::size_t row = 0; row < row_count; ++row) {
            (*out_codes)[row * residual_pq.num_chunks + chunk] =
                static_cast<std::uint8_t>(closest_centers[row]);
        }
    }
}

struct ErrorStats {
    double sum = 0.0;
    float min = std::numeric_limits<float>::infinity();
    float max = 0.0f;
};

ErrorStats ReadExistingErrorStats(const std::filesystem::path& path, std::size_t row_count) {
    ErrorStats stats;
    if (row_count == 0) {
        return stats;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open " + path.string());
    }
    in.seekg(static_cast<std::streamoff>(sizeof(std::int32_t) * 2), std::ios::beg);
    constexpr std::size_t kChunkRows = 1U << 20;
    std::vector<float> buffer(std::min(row_count, kChunkRows), 0.0f);
    std::size_t rows_read = 0;
    while (rows_read < row_count) {
        const std::size_t chunk = std::min(buffer.size(), row_count - rows_read);
        in.read(reinterpret_cast<char*>(buffer.data()),
                static_cast<std::streamsize>(chunk * sizeof(float)));
        if (!in.good()) {
            throw std::runtime_error("Short read while scanning existing error file: " +
                                     path.string());
        }
        for (std::size_t i = 0; i < chunk; ++i) {
            const float value = buffer[i];
            stats.sum += value;
            stats.min = std::min(stats.min, value);
            stats.max = std::max(stats.max, value);
        }
        rows_read += chunk;
    }
    return stats;
}

void WriteHeader(std::ofstream& out, std::int32_t rows, std::int32_t cols) {
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
}

int Main(const Args& args) {
    const FbinMetadata data_meta = ReadFbinMetadata(args.data_bin);
    const PqPivotData base_pq = LoadPqPivots(args.base_pq_pivots);
    if (base_pq.ndims != data_meta.cols) {
        throw std::runtime_error("Base PQ dim does not match data dim.");
    }
    if (base_pq.num_chunks != args.num_chunks) {
        throw std::runtime_error("Base PQ num_chunks does not match requested num_chunks.");
    }

    const auto [base_code_rows, base_code_cols] = ReadBinMetadata(args.base_pq_codes, 0);
    if (base_code_rows != data_meta.rows || base_code_cols != args.num_chunks) {
        throw std::runtime_error("Base PQ code shape does not match data.");
    }

    std::size_t resume_row_begin = 0;
    bool reuse_existing_pivots = false;
    if (args.resume) {
        const OutputFileState code_state = InspectOutputFile(args.residual_pq_codes, sizeof(std::uint8_t));
        const OutputFileState error_state = InspectOutputFile(args.residual_pq_error, sizeof(float));
        if (code_state.exists != error_state.exists) {
            throw std::runtime_error(
                "Resume requires both residual code and error files to exist together.");
        }
        if (code_state.exists) {
            if (code_state.rows != data_meta.rows || code_state.cols != args.num_chunks) {
                throw std::runtime_error("Residual code output header does not match requested layout.");
            }
            if (error_state.rows != data_meta.rows || error_state.cols != 1) {
                throw std::runtime_error("Residual error output header does not match requested layout.");
            }
            if (code_state.encoded_rows != error_state.encoded_rows) {
                throw std::runtime_error("Residual code/error outputs have different encoded row counts.");
            }
            if (code_state.encoded_rows > data_meta.rows) {
                throw std::runtime_error("Residual output encoded row count exceeds dataset rows.");
            }
            resume_row_begin = code_state.encoded_rows;
        }
        reuse_existing_pivots = std::filesystem::exists(args.residual_pq_pivots);
        if (resume_row_begin != 0 && !reuse_existing_pivots) {
            throw std::runtime_error("Resume from partial outputs requires an existing residual pivot file.");
        }
        if (resume_row_begin == data_meta.rows) {
            std::cout << "[topoanns_build_residual_pq] outputs already complete"
                      << " rows=" << data_meta.rows << std::endl;
            return 0;
        }
    }

    if (!reuse_existing_pivots) {
        const std::size_t train_points = std::min(args.train_points, data_meta.rows);
        std::vector<float> train_vectors;
        std::vector<std::uint8_t> train_base_codes;
        std::vector<float> train_base_recon;
        ReadFloatRows(args.data_bin, 0, train_points, data_meta.cols, &train_vectors);
        ReadCodeRows(args.base_pq_codes, 0, train_points, args.num_chunks, &train_base_codes);
        ReconstructVectors(base_pq, train_base_codes, train_points, &train_base_recon);
        std::vector<float> train_residuals = ComputeResiduals(train_vectors, train_base_recon);

        std::cout << "[topoanns_build_residual_pq] train_points=" << train_points
                  << " dim=" << data_meta.cols
                  << " num_chunks=" << args.num_chunks << std::endl;
        if (std::filesystem::exists(args.residual_pq_pivots)) {
            std::filesystem::remove(args.residual_pq_pivots);
        }
        const int train_rc = diskann::generate_pq_pivots(
            train_residuals.data(), train_points, static_cast<std::uint32_t>(data_meta.cols),
            static_cast<std::uint32_t>(kNumPqCentroids), args.num_chunks, kResidualPqKmeansIters,
            args.residual_pq_pivots.string(), false);
        if (train_rc != 0) {
            throw std::runtime_error("generate_pq_pivots returned non-zero.");
        }
    } else {
        std::cout << "[topoanns_build_residual_pq] reuse_existing_pivots=1"
                  << " resume_row_begin=" << resume_row_begin
                  << " dim=" << data_meta.cols
                  << " num_chunks=" << args.num_chunks << std::endl;
    }

    const PqPivotData residual_pq = LoadPqPivots(args.residual_pq_pivots);
    if (residual_pq.ndims != data_meta.cols || residual_pq.num_chunks != args.num_chunks) {
        throw std::runtime_error("Residual PQ pivot metadata does not match requested layout.");
    }

    std::ofstream codes_out;
    std::ofstream error_out;
    if (resume_row_begin == 0) {
        codes_out.open(args.residual_pq_codes, std::ios::binary | std::ios::trunc);
        error_out.open(args.residual_pq_error, std::ios::binary | std::ios::trunc);
    } else {
        codes_out.open(args.residual_pq_codes, std::ios::binary | std::ios::in | std::ios::out);
        error_out.open(args.residual_pq_error, std::ios::binary | std::ios::in | std::ios::out);
    }
    if (!codes_out.is_open() || !error_out.is_open()) {
        throw std::runtime_error("Failed to open residual PQ outputs.");
    }
    if (resume_row_begin == 0) {
        WriteHeader(codes_out, static_cast<std::int32_t>(data_meta.rows),
                    static_cast<std::int32_t>(args.num_chunks));
        WriteHeader(error_out, static_cast<std::int32_t>(data_meta.rows), 1);
    } else {
        const std::streamoff code_offset = static_cast<std::streamoff>(sizeof(std::int32_t) * 2 +
                                                                       resume_row_begin *
                                                                           args.num_chunks);
        const std::streamoff error_offset = static_cast<std::streamoff>(sizeof(std::int32_t) * 2 +
                                                                        resume_row_begin *
                                                                            sizeof(float));
        codes_out.seekp(code_offset, std::ios::beg);
        error_out.seekp(error_offset, std::ios::beg);
    }

    double error_sum = 0.0;
    float error_min = std::numeric_limits<float>::infinity();
    float error_max = 0.0f;
    std::size_t total_points = resume_row_begin;
    if (resume_row_begin != 0) {
        const ErrorStats existing_stats =
            ReadExistingErrorStats(args.residual_pq_error, resume_row_begin);
        error_sum = existing_stats.sum;
        error_min = existing_stats.min;
        error_max = existing_stats.max;
    }

    std::vector<float> block_vectors;
    std::vector<std::uint8_t> block_base_codes;
    std::vector<float> block_base_recon;
    std::vector<float> block_residuals;
    std::vector<std::uint8_t> block_residual_codes;
    std::vector<float> block_residual_recon;
    std::vector<float> block_errors;
    for (std::size_t row_begin = resume_row_begin; row_begin < data_meta.rows;
         row_begin += args.block_points) {
        const std::size_t row_count = std::min(args.block_points, data_meta.rows - row_begin);
        ReadFloatRows(args.data_bin, row_begin, row_count, data_meta.cols, &block_vectors);
        ReadCodeRows(args.base_pq_codes, row_begin, row_count, args.num_chunks, &block_base_codes);
        ReconstructVectors(base_pq, block_base_codes, row_count, &block_base_recon);
        block_residuals = ComputeResiduals(block_vectors, block_base_recon);

        EncodeResidualBlock(residual_pq, block_residuals, row_count, &block_residual_codes);
        if (block_residual_codes.size() != row_count * args.num_chunks) {
            throw std::runtime_error("Residual PQ code size mismatch.");
        }

        ReconstructVectors(residual_pq, block_residual_codes, row_count, &block_residual_recon);
        block_errors.assign(row_count, 0.0f);
        for (std::size_t row = 0; row < row_count; ++row) {
            double accum = 0.0;
            const float* residual_ptr = block_residuals.data() + row * data_meta.cols;
            const float* recon_ptr = block_residual_recon.data() + row * data_meta.cols;
            for (std::size_t dim = 0; dim < data_meta.cols; ++dim) {
                const double diff =
                    static_cast<double>(residual_ptr[dim]) - static_cast<double>(recon_ptr[dim]);
                accum += diff * diff;
            }
            const float error = static_cast<float>(std::sqrt(accum));
            block_errors[row] = error;
            error_sum += error;
            error_min = std::min(error_min, error);
            error_max = std::max(error_max, error);
        }

        codes_out.write(reinterpret_cast<const char*>(block_residual_codes.data()),
                        static_cast<std::streamsize>(block_residual_codes.size()));
        error_out.write(reinterpret_cast<const char*>(block_errors.data()),
                        static_cast<std::streamsize>(block_errors.size() * sizeof(float)));
        total_points += row_count;

        std::cout << "[topoanns_build_residual_pq] encoded_rows=" << total_points
                  << "/" << data_meta.rows << std::endl;
    }

    codes_out.close();
    error_out.close();

    std::cout << "[topoanns_build_residual_pq] residual_codes=" << args.residual_pq_codes
              << " residual_error=" << args.residual_pq_error
              << " error_min=" << error_min
              << " error_max=" << error_max
              << " error_avg=" << (total_points == 0 ? 0.0 : error_sum / total_points)
              << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(ParseArgs(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_build_residual_pq] " << e.what() << std::endl;
        return 1;
    }
}
