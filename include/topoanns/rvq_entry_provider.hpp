#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "topoanns/device_entry_batch.hpp"
#include "topoanns/entry_provider.hpp"

namespace topoanns {

struct RvqEntryProfile {
    double total_ms = 0.0;
    double query_upload_ms = 0.0;
    double search_kernel_ms = 0.0;
    double entry_gather_ms = 0.0;
};

class RvqModel {
public:
    RvqModel();
    ~RvqModel();
    RvqModel(RvqModel&&) noexcept;
    RvqModel& operator=(RvqModel&&) noexcept;

    RvqModel(const RvqModel&) = delete;
    RvqModel& operator=(const RvqModel&) = delete;

    static RvqModel Load(const std::filesystem::path& model_path);

    std::uint32_t dim() const;
    std::uint32_t coarse_centroids() const;
    std::uint32_t fine_centroids() const;
    void WarmUp() const;

    std::vector<std::vector<std::uint32_t>> ComputeFloat32EntryPoints(
        const std::vector<float>& queries,
        std::size_t num_queries,
        std::size_t entry_count,
        std::uint32_t fallback_entry = 0,
        RvqEntryProfile* out_profile = nullptr) const;

    DeviceEntryBatch ComputeFloat32DeviceEntryBatch(
        const std::vector<float>& queries,
        std::size_t num_queries,
        std::size_t entry_count,
        std::uint32_t fallback_entry = 0,
        RvqEntryProfile* out_profile = nullptr) const;

private:
    struct Impl;

    explicit RvqModel(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

class RvqEntryProvider final : public EntryProvider {
public:
    explicit RvqEntryProvider(std::vector<std::vector<std::uint32_t>> entries_by_query);

    static RvqEntryProvider FromFloatQueries(const RvqModel& model,
                                             const std::vector<float>& queries,
                                             std::size_t num_queries,
                                             std::size_t entry_count,
                                             std::uint32_t fallback_entry = 0,
                                             RvqEntryProfile* out_profile = nullptr);

    std::vector<std::uint32_t> GetEntryPoints(std::size_t query_id) const override;
    std::size_t num_queries() const { return entries_by_query_.size(); }

private:
    std::vector<std::vector<std::uint32_t>> entries_by_query_;
};

}  // namespace topoanns
