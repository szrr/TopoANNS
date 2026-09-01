#include "topoanns/search_stop_condition.hpp"

#include <iostream>
#include <limits>
#include <vector>

#include "test_support.hpp"

int main() {
    using topoanns::RankedCandidate;
    using topoanns::SearchStopCondition;
    using topoanns::kInvalidNodeId;

    std::vector<RankedCandidate> candidates = {
        {0.10f, 100U, true},
        {0.20f, 200U, true},
        {0.30f, 300U, true},
        {0.40f, 400U, false},
        {0.50f, 500U, false},
    };

    TOPOANNS_ASSERT_EQ(SearchStopCondition::CountValid(candidates), 5ULL);
    TOPOANNS_ASSERT_EQ(SearchStopCondition::BestUnexpandedRank(candidates), 3ULL);
    TOPOANNS_ASSERT_TRUE(SearchStopCondition::TopLHasNoBetterUnexpanded(candidates, 3));
    TOPOANNS_ASSERT_TRUE(!SearchStopCondition::TopLHasNoBetterUnexpanded(candidates, 4));

    candidates[0].expanded = false;
    TOPOANNS_ASSERT_EQ(SearchStopCondition::BestUnexpandedRank(candidates), 0ULL);
    TOPOANNS_ASSERT_TRUE(!SearchStopCondition::TopLHasNoBetterUnexpanded(candidates, 3));

    candidates = {
        {0.10f, 100U, true},
        {0.20f, 200U, true},
        {0.30f, kInvalidNodeId, false},
        {0.40f, kInvalidNodeId, false},
    };
    TOPOANNS_ASSERT_EQ(SearchStopCondition::CountValid(candidates), 2ULL);
    TOPOANNS_ASSERT_TRUE(!SearchStopCondition::TopLHasNoBetterUnexpanded(candidates, 3));

    candidates = {
        {0.10f, 100U, true},
        {0.20f, 200U, true},
        {0.30f, 300U, true},
    };
    TOPOANNS_ASSERT_EQ(
        SearchStopCondition::BestUnexpandedRank(candidates),
        std::numeric_limits<std::size_t>::max());
    TOPOANNS_ASSERT_TRUE(SearchStopCondition::TopLHasNoBetterUnexpanded(candidates, 3));

    std::cout << "test_search_stop_condition passed" << std::endl;
    return 0;
}
