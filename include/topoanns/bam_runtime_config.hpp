#pragma once

#include "topoanns/bam_vector_provider.hpp"

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace topoanns {

struct BamRuntimeConfig {
    std::filesystem::path source_path;
    std::filesystem::path controller_path = "/dev/libnvm0";
    std::uint32_t cuda_device = 0;
    std::uint32_t nvm_namespace = 1;
    std::size_t page_cache_size_bytes = 16ULL << 20;
    std::size_t queue_depth = 256;
    std::size_t num_queues = 8;
};

struct BamRuntimeConfigOverrides {
    std::optional<std::filesystem::path> controller_path;
    std::optional<std::uint32_t> cuda_device;
    std::optional<std::uint32_t> nvm_namespace;
    std::optional<std::size_t> page_cache_size_bytes;
    std::optional<std::size_t> queue_depth;
    std::optional<std::size_t> num_queues;
};

inline std::string TrimBamConfig(const std::string& text) {
    std::size_t start = 0;
    while (start < text.size() &&
           std::isspace(static_cast<unsigned char>(text[start])) != 0) {
        ++start;
    }
    std::size_t end = text.size();
    while (end > start &&
           std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) {
        --end;
    }
    return text.substr(start, end - start);
}

inline std::optional<std::filesystem::path> DetectDefaultBamRuntimeConfigPath() {
    if (const char* env = std::getenv("TOPOANNS_BAM_CONFIG")) {
        if (*env == '\0') {
            throw std::runtime_error(
                "TOPOANNS_BAM_CONFIG is set but empty. Point it to a BAM config file.");
        }
        return std::filesystem::path(env);
    }

    std::error_code ec;
    const std::filesystem::path cwd_candidate =
        std::filesystem::current_path(ec) / "config" / "bam_runtime.conf";
    if (!ec && std::filesystem::exists(cwd_candidate)) {
        return cwd_candidate;
    }

    ec.clear();
    const std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec) {
        const std::filesystem::path exe_candidate =
            exe.parent_path().parent_path() / "config" / "bam_runtime.conf";
        if (std::filesystem::exists(exe_candidate)) {
            return exe_candidate;
        }
    }
    return std::nullopt;
}

inline std::filesystem::path ResolveBamRuntimeConfigPath(
    const std::optional<std::filesystem::path>& config_path_override) {
    if (config_path_override.has_value()) {
        return *config_path_override;
    }
    const auto detected = DetectDefaultBamRuntimeConfigPath();
    if (detected.has_value()) {
        return *detected;
    }
    throw std::runtime_error(
        "Missing BAM runtime config. Create config/bam_runtime.conf or pass "
        "--bam-config-path <path>. Refusing to fall back to a hardcoded /dev/libnvm0.");
}

inline BamRuntimeConfig LoadBamRuntimeConfig(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Failed to open BAM runtime config: " + path.string());
    }

    BamRuntimeConfig config;
    config.source_path = path;
    std::string line;
    std::size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const std::size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.erase(comment);
        }
        line = TrimBamConfig(line);
        if (line.empty()) {
            continue;
        }

        const std::size_t equals = line.find('=');
        if (equals == std::string::npos) {
            throw std::runtime_error("Invalid BAM runtime config line " +
                                     std::to_string(line_no) + " in " + path.string() +
                                     ": expected key=value.");
        }

        const std::string key = TrimBamConfig(line.substr(0, equals));
        const std::string value = TrimBamConfig(line.substr(equals + 1));
        if (key.empty() || value.empty()) {
            throw std::runtime_error("Invalid BAM runtime config line " +
                                     std::to_string(line_no) + " in " + path.string() +
                                     ": empty key or value.");
        }

        if (key == "controller_path") {
            config.controller_path = value;
        } else if (key == "cuda_device") {
            config.cuda_device = static_cast<std::uint32_t>(std::stoul(value));
        } else if (key == "nvm_namespace") {
            config.nvm_namespace = static_cast<std::uint32_t>(std::stoul(value));
        } else if (key == "page_cache_size_bytes" || key == "page_cache_bytes") {
            config.page_cache_size_bytes = std::stoull(value);
        } else if (key == "queue_depth") {
            config.queue_depth = std::stoull(value);
        } else if (key == "num_queues") {
            config.num_queues = std::stoull(value);
        } else {
            throw std::runtime_error("Unknown BAM runtime config key \"" + key +
                                     "\" in " + path.string() + ".");
        }
    }

    return config;
}

inline BamRuntimeConfig ResolveBamRuntimeConfig(
    const std::optional<std::filesystem::path>& config_path_override,
    const BamRuntimeConfigOverrides& overrides,
    bool allow_controller_path_override) {
    BamRuntimeConfig config =
        LoadBamRuntimeConfig(ResolveBamRuntimeConfigPath(config_path_override));

    if (overrides.controller_path.has_value()) {
        if (!allow_controller_path_override &&
            overrides.controller_path->string() != config.controller_path.string()) {
            throw std::runtime_error(
                "Explicit BAM controller path \"" + overrides.controller_path->string() +
                "\" conflicts with BAM runtime config \"" + config.controller_path.string() +
                "\" from " + config.source_path.string() +
                ". Update the shared config or pass --allow-bam-controller-override if this "
                "mismatch is intentional.");
        }
        config.controller_path = *overrides.controller_path;
    }
    if (overrides.cuda_device.has_value()) {
        config.cuda_device = *overrides.cuda_device;
    }
    if (overrides.nvm_namespace.has_value()) {
        config.nvm_namespace = *overrides.nvm_namespace;
    }
    if (overrides.page_cache_size_bytes.has_value()) {
        config.page_cache_size_bytes = *overrides.page_cache_size_bytes;
    }
    if (overrides.queue_depth.has_value()) {
        config.queue_depth = *overrides.queue_depth;
    }
    if (overrides.num_queues.has_value()) {
        config.num_queues = *overrides.num_queues;
    }
    return config;
}

inline void ApplyBamRuntimeConfig(const BamRuntimeConfig& config,
                                  BamVectorProviderOptions* options) {
    options->controller_path = config.controller_path;
    options->cuda_device = config.cuda_device;
    options->nvm_namespace = config.nvm_namespace;
    options->page_cache_size_bytes = config.page_cache_size_bytes;
    options->queue_depth = config.queue_depth;
    options->num_queues = config.num_queues;
}

}  // namespace topoanns
