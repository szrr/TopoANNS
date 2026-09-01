#include "topoanns/search_stop_condition.hpp"

#include <limits>

namespace topoanns {

std::size_t SearchStopCondition::CountValid(
    const std::vector<RankedCandidate>& candidates) {
    std::size_t valid = 0;
    for (const RankedCandidate& candidate : candidates) {
        valid += candidate.valid() ? 1U : 0U;
    }
    return valid;
}

std::size_t SearchStopCondition::BestUnexpandedRank(
    const std::vector<RankedCandidate>& candidates) {
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (candidates[i].valid() && !candidates[i].expanded) {
            return i;
        }
    }
    return std::numeric_limits<std::size_t>::max();
}

bool SearchStopCondition::TopLHasNoBetterUnexpanded(
    const std::vector<RankedCandidate>& candidates,
    std::size_t top_l) {
    if (top_l == 0) {
        return true;
    }
    if (CountValid(candidates) < top_l) {
        return false;
    }
    const std::size_t best_unexpanded_rank = BestUnexpandedRank(candidates);
    return best_unexpanded_rank >= top_l;
}

}  // namespace topoanns
