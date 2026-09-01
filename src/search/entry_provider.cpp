#include "topoanns/entry_provider.hpp"

#include <stdexcept>

namespace topoanns {

FixedEntryProvider::FixedEntryProvider(std::vector<std::uint32_t> entries)
    : entries_(std::move(entries)) {}

std::vector<std::uint32_t> FixedEntryProvider::GetEntryPoints(std::size_t /*query_id*/) const {
    return entries_;
}

PerQueryEntryProvider::PerQueryEntryProvider(
    std::vector<std::vector<std::uint32_t>> entries_by_query)
    : entries_by_query_(std::move(entries_by_query)) {}

std::vector<std::uint32_t> PerQueryEntryProvider::GetEntryPoints(
    std::size_t query_id) const {
    if (query_id >= entries_by_query_.size()) {
        throw std::runtime_error(BuildErrorMessage(
            "PerQueryEntryProvider::GetEntryPoints", "query_id is out of range."));
    }
    return entries_by_query_[query_id];
}

}  // namespace topoanns
