#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "topoanns/common.hpp"

namespace topoanns {

class EntryProvider {
public:
    virtual ~EntryProvider() = default;
    virtual std::vector<std::uint32_t> GetEntryPoints(std::size_t query_id) const = 0;
};

class FixedEntryProvider final : public EntryProvider {
public:
    explicit FixedEntryProvider(std::vector<std::uint32_t> entries);
    std::vector<std::uint32_t> GetEntryPoints(std::size_t query_id) const override;

private:
    std::vector<std::uint32_t> entries_;
};

class PerQueryEntryProvider final : public EntryProvider {
public:
    explicit PerQueryEntryProvider(std::vector<std::vector<std::uint32_t>> entries_by_query);
    std::vector<std::uint32_t> GetEntryPoints(std::size_t query_id) const override;

private:
    std::vector<std::vector<std::uint32_t>> entries_by_query_;
};

}  // namespace topoanns
