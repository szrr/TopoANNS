#include "topoanns/entry_provider.hpp"

#include <exception>
#include <iostream>
#include <vector>

#include "test_support.hpp"

int main() {
    using topoanns::FixedEntryProvider;
    using topoanns::PerQueryEntryProvider;

    FixedEntryProvider fixed({7U, 11U, 13U});
    const auto fixed_q0 = fixed.GetEntryPoints(0);
    const auto fixed_q9 = fixed.GetEntryPoints(9);
    TOPOANNS_ASSERT_EQ(fixed_q0.size(), 3ULL);
    TOPOANNS_ASSERT_EQ(fixed_q0[0], 7U);
    TOPOANNS_ASSERT_EQ(fixed_q9[2], 13U);

    PerQueryEntryProvider per_query({
        {1U, 2U},
        {3U},
        {4U, 5U, 6U},
    });
    TOPOANNS_ASSERT_EQ(per_query.GetEntryPoints(0).size(), 2ULL);
    TOPOANNS_ASSERT_EQ(per_query.GetEntryPoints(1)[0], 3U);
    TOPOANNS_ASSERT_EQ(per_query.GetEntryPoints(2)[2], 6U);

    bool threw = false;
    try {
        (void)per_query.GetEntryPoints(3);
    } catch (const std::exception&) {
        threw = true;
    }
    TOPOANNS_ASSERT_TRUE(threw);

    std::cout << "test_entry_provider passed" << std::endl;
    return 0;
}
