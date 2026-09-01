#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace topoanns::test {

[[noreturn]] inline void Fail(const char* file, int line,
                              const std::string& message) {
    std::cerr << file << ":" << line << ": " << message << std::endl;
    std::exit(1);
}

template <typename A, typename B>
inline void AssertEqual(const A& actual, const B& expected,
                        const char* actual_expr, const char* expected_expr,
                        const char* file, int line) {
    if (!(actual == expected)) {
        std::ostringstream oss;
        oss << "Assertion failed: " << actual_expr << " == " << expected_expr
            << " (actual=" << actual << ", expected=" << expected << ")";
        Fail(file, line, oss.str());
    }
}

inline void AssertTrue(bool condition, const char* expr,
                       const char* file, int line) {
    if (!condition) {
        Fail(file, line, std::string("Assertion failed: ") + expr);
    }
}

}  // namespace topoanns::test

#define TOPOANNS_ASSERT_TRUE(expr) \
    ::topoanns::test::AssertTrue((expr), #expr, __FILE__, __LINE__)

#define TOPOANNS_ASSERT_EQ(actual, expected) \
    ::topoanns::test::AssertEqual((actual), (expected), #actual, #expected, __FILE__, __LINE__)
