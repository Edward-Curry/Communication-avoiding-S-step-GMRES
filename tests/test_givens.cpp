/**
 * @file tests/test_givens.cpp
 * @brief Tests Givens rotation generation and application.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "common/givens.hpp"

#include <cassert>
#include <cmath>
#include <print>

/**
 * @brief Runs the Givens rotation test.
 * @return Zero when all assertions pass.
 */
int main()
{
    gmres::Scalar a = 3.0;
    gmres::Scalar b = 4.0;

    gmres::Scalar c = 0.0;
    gmres::Scalar s = 0.0;

    gmres::generate_givens(a, b, c, s);

    assert(std::abs(c - 0.6) < 1e-12);
    assert(std::abs(s - 0.8) < 1e-12);

    gmres::apply_givens(c, s, a, b);

    assert(std::abs(a - 5.0) < 1e-12);
    assert(std::abs(b - 0.0) < 1e-12);

    gmres::Scalar a2 = 7.0;
    gmres::Scalar b2 = 0.0;

    gmres::generate_givens(a2, b2, c, s);

    assert(std::abs(c - 1.0) < 1e-12);
    assert(std::abs(s - 0.0) < 1e-12);

    gmres::apply_givens(c, s, a2, b2);

    assert(std::abs(a2 - 7.0) < 1e-12);
    assert(std::abs(b2 - 0.0) < 1e-12);

    std::println("Givens rotation test passed.");

    return 0;
}
