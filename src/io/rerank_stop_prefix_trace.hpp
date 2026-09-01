#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace topoanns::detail {

constexpr std::uint32_t kRerankStopPrefixTraceMagic = 0x52535031U;

struct RerankStopPrefixTraceHeader {
    std::uint32_t magic = kRerankStopPrefixTraceMagic;
    std::uint32_t budget_top_n = 0U;
    std::uint32_t num_queries = 0U;
};

inline std::string NonEmptyEnvironmentValue(const char* name) {
    const char* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

inline void AppendRerankStopPrefixTrace(
    const std::filesystem::path& path,
    std::size_t budget_top_n,
    const std::vector<std::uint32_t>& prefixes) {
    static std::mutex mutex;
    static std::unordered_map<std::string, bool> initialized;
    const std::string key = path.string();
    std::lock_guard<std::mutex> lock(mutex);
    const bool append = initialized[key];
    std::ofstream out(path, std::ios::binary | (append ? std::ios::app : std::ios::trunc));
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open rerank stop-prefix trace for writing: " + key);
    }
    if (budget_top_n > std::numeric_limits<std::uint32_t>::max() ||
        prefixes.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Rerank stop-prefix trace dimensions exceed uint32 range.");
    }
    const RerankStopPrefixTraceHeader header{
        kRerankStopPrefixTraceMagic, static_cast<std::uint32_t>(budget_top_n),
        static_cast<std::uint32_t>(prefixes.size())};
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    out.write(reinterpret_cast<const char*>(prefixes.data()),
              static_cast<std::streamsize>(prefixes.size() * sizeof(std::uint32_t)));
    if (!out) {
        throw std::runtime_error("Failed to write rerank stop-prefix trace: " + key);
    }
    initialized[key] = true;
}

inline std::vector<std::uint32_t> ReadRerankStopPrefixTrace(
    const std::filesystem::path& path,
    std::size_t expected_budget_top_n,
    std::size_t expected_num_queries) {
    static std::mutex mutex;
    static std::unordered_map<std::string, std::unique_ptr<std::ifstream>> streams;
    const std::string key = path.string();
    std::lock_guard<std::mutex> lock(mutex);
    auto& stream = streams[key];
    if (!stream) {
        stream = std::make_unique<std::ifstream>(path, std::ios::binary);
    }
    if (!stream->is_open()) {
        throw std::runtime_error("Failed to open rerank stop-prefix trace for reading: " + key);
    }
    RerankStopPrefixTraceHeader header{};
    stream->read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!*stream || header.magic != kRerankStopPrefixTraceMagic ||
        header.budget_top_n != expected_budget_top_n ||
        header.num_queries != expected_num_queries) {
        throw std::runtime_error("Rerank stop-prefix trace mismatch or premature EOF: " + key);
    }
    std::vector<std::uint32_t> prefixes(expected_num_queries);
    stream->read(reinterpret_cast<char*>(prefixes.data()),
                 static_cast<std::streamsize>(prefixes.size() * sizeof(std::uint32_t)));
    if (!*stream) {
        throw std::runtime_error("Truncated rerank stop-prefix trace payload: " + key);
    }
    return prefixes;
}

}  // namespace topoanns::detail
