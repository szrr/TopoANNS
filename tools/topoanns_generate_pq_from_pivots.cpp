#include <immintrin.h>

#include "pq.h"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint32_t kNumPqCentroids = 256U;
constexpr std::uint32_t kPqTrainingIterations = 15U;

struct Args {
    std::filesystem::path data_bin;
    std::filesystem::path train_data_bin;
    std::filesystem::path pq_pivots;
    std::filesystem::path pq_codes;
    std::uint32_t num_chunks = 0;
    bool make_zero_mean = false;
};

struct FloatBin {
    std::uint32_t rows = 0;
    std::uint32_t cols = 0;
    std::vector<float> values;
};

[[noreturn]] void Usage() {
    std::cerr
        << "Usage: topoanns_generate_pq_from_pivots"
        << " --data-bin <path>"
        << " --pq-pivots <path>"
        << " --pq-codes <path>"
        << " --num-chunks <count>"
        << " [--train-data-bin <path> --make-zero-mean]"
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
        } else if (flag == "--train-data-bin") {
            args.train_data_bin = read_value("--train-data-bin");
        } else if (flag == "--pq-pivots") {
            args.pq_pivots = read_value("--pq-pivots");
        } else if (flag == "--pq-codes") {
            args.pq_codes = read_value("--pq-codes");
        } else if (flag == "--num-chunks") {
            args.num_chunks = static_cast<std::uint32_t>(std::stoul(read_value("--num-chunks")));
        } else if (flag == "--make-zero-mean") {
            args.make_zero_mean = true;
        } else {
            throw std::runtime_error("Unknown flag: " + std::string(flag));
        }
    }
    if (args.data_bin.empty() || args.pq_pivots.empty() || args.pq_codes.empty() ||
        args.num_chunks == 0) {
        Usage();
    }
    return args;
}

FloatBin LoadFloatBin(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open training data: " + path.string());
    }
    FloatBin data;
    input.read(reinterpret_cast<char*>(&data.rows), sizeof(data.rows));
    input.read(reinterpret_cast<char*>(&data.cols), sizeof(data.cols));
    if (!input || data.rows == 0 || data.cols == 0) {
        throw std::runtime_error("Invalid training data header: " + path.string());
    }
    data.values.resize(static_cast<std::size_t>(data.rows) * data.cols);
    input.read(reinterpret_cast<char*>(data.values.data()),
               static_cast<std::streamsize>(data.values.size() * sizeof(float)));
    if (!input) {
        throw std::runtime_error("Short read in training data: " + path.string());
    }
    return data;
}

int Main(const Args& args) {
    std::cout << "[topoanns_generate_pq_from_pivots] data_bin=" << args.data_bin << std::endl;
    std::cout << "[topoanns_generate_pq_from_pivots] pq_pivots=" << args.pq_pivots << std::endl;
    std::cout << "[topoanns_generate_pq_from_pivots] pq_codes=" << args.pq_codes << std::endl;
    std::cout << "[topoanns_generate_pq_from_pivots] num_chunks=" << args.num_chunks << std::endl;

    if (!std::filesystem::exists(args.pq_pivots)) {
        if (args.train_data_bin.empty()) {
            throw std::runtime_error(
                "PQ pivots are missing; pass --train-data-bin to train them first.");
        }
        const FloatBin training = LoadFloatBin(args.train_data_bin);
        if (args.num_chunks > training.cols) {
            throw std::runtime_error("num-chunks must not exceed the vector dimension.");
        }
        std::cout << "[topoanns_generate_pq_from_pivots] training_rows=" << training.rows
                  << " training_dim=" << training.cols
                  << " make_zero_mean=" << (args.make_zero_mean ? 1 : 0) << std::endl;
        const int train_rc = diskann::generate_pq_pivots(
            training.values.data(), training.rows, training.cols, kNumPqCentroids,
            args.num_chunks, kPqTrainingIterations, args.pq_pivots.string(),
            args.make_zero_mean);
        if (train_rc != 0) {
            throw std::runtime_error("generate_pq_pivots returned non-zero.");
        }
    }

    const int rc = diskann::generate_pq_data_from_pivots<float>(
        args.data_bin.string(), kNumPqCentroids, args.num_chunks,
        args.pq_pivots.string(), args.pq_codes.string(), false);
    if (rc != 0) {
        throw std::runtime_error("generate_pq_data_from_pivots returned non-zero.");
    }

    std::cout << "[topoanns_generate_pq_from_pivots] completed" << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return Main(ParseArgs(argc, argv));
    } catch (const std::exception& e) {
        std::cerr << "[topoanns_generate_pq_from_pivots] " << e.what() << std::endl;
        return 1;
    }
}
