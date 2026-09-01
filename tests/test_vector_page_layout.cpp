#include "topoanns/vector_page_layout.hpp"

#include <exception>
#include <iostream>

#include "test_support.hpp"

int main() {
    using topoanns::ScalarKind;
    using topoanns::VectorPageLayout;

    const VectorPageLayout float_layout =
        VectorPageLayout::Create(128, ScalarKind::kFloat32);
    TOPOANNS_ASSERT_EQ(float_layout.vector_bytes(), 512ULL);
    TOPOANNS_ASSERT_EQ(float_layout.vectors_per_page(), 8ULL);
    TOPOANNS_ASSERT_EQ(float_layout.payload_bytes_per_page(), 4096ULL);
    TOPOANNS_ASSERT_EQ(float_layout.PageCountForVectors(17), 3ULL);

    const auto node0 = float_layout.Resolve(0);
    TOPOANNS_ASSERT_EQ(node0.page_id, 0ULL);
    TOPOANNS_ASSERT_EQ(node0.slot_id, 0U);
    TOPOANNS_ASSERT_EQ(node0.byte_offset, 4096ULL);

    const auto node7 = float_layout.Resolve(7);
    TOPOANNS_ASSERT_EQ(node7.page_id, 0ULL);
    TOPOANNS_ASSERT_EQ(node7.slot_id, 7U);
    TOPOANNS_ASSERT_EQ(node7.byte_offset, 4096ULL + 7ULL * 512ULL);

    const auto node8 = float_layout.Resolve(8);
    TOPOANNS_ASSERT_EQ(node8.page_id, 1ULL);
    TOPOANNS_ASSERT_EQ(node8.slot_id, 0U);
    TOPOANNS_ASSERT_EQ(node8.byte_offset, 4096ULL + 4096ULL);

    const VectorPageLayout byte_layout =
        VectorPageLayout::Create(100, ScalarKind::kUint8);
    TOPOANNS_ASSERT_EQ(byte_layout.vector_bytes(), 100ULL);
    TOPOANNS_ASSERT_EQ(byte_layout.vectors_per_page(), 40ULL);
    TOPOANNS_ASSERT_TRUE(byte_layout.payload_bytes_per_page() <=
                         byte_layout.page_size_bytes());

    bool threw = false;
    try {
        (void)VectorPageLayout::Create(2048, ScalarKind::kFloat32);
    } catch (const std::exception&) {
        threw = true;
    }
    TOPOANNS_ASSERT_TRUE(threw);

    std::cout << "test_vector_page_layout passed" << std::endl;
    return 0;
}
