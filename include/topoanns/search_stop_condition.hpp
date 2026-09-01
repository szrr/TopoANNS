#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "topoanns/common.hpp"

namespace topoanns {

struct RankedCandidate {
    float distance = 0.0f;
    std::uint32_t node_id = kInvalidNodeId;
    bool expanded = false;

    bool valid() const { return node_id != kInvalidNodeId; }
};

class SearchStopCondition {
public:
    static std::size_t CountValid(const std::vector<RankedCandidate>& candidates);
    static std::size_t BestUnexpandedRank(const std::vector<RankedCandidate>& candidates);
    static bool TopLHasNoBetterUnexpanded(const std::vector<RankedCandidate>& candidates,
                                         std::size_t top_l);
};

}  // namespace topoanns
