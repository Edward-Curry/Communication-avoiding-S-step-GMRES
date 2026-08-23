/**
 * @file tests/test_arnoldi_seq.cpp
 * @brief Tests sequential Arnoldi factorization.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "sequential/arnoldi_seq.hpp"

#include "common/sparse_matrix.hpp"
#include "common/vector_ops.hpp"

#include <cassert>
#include <cmath>
#include <print>
#include <vector>

/**
 * @brief Runs the sequential Arnoldi test.
 * @return Zero when all assertions pass.
 */
int main()
{
    gmres::Vector values = {
        1.0, 1.0,
        1.0, 1.0,
        1.0
    };

    std::vector<gmres::Index> col_indices = {
        0, 1,
        1, 2,
        2
    };

    std::vector<gmres::Index> row_ptr = {
        0, 2, 4, 5
    };

    gmres::SparseMatrixCSR A(3, 3, values, col_indices, row_ptr);

    gmres::Vector r0 = {1.0, 1.0, 1.0};

    gmres::ArnoldiResult result = gmres::arnoldi_seq(A, r0, 2);

    assert(std::abs(result.beta - std::sqrt(3.0)) < 1e-12);

    assert(result.V.size() == 3);
    assert(result.H.size() == 3);
    assert(result.H[0].size() == 2);

    assert(std::abs(gmres::norm2(result.V[0]) - 1.0) < 1e-12);
    assert(std::abs(gmres::norm2(result.V[1]) - 1.0) < 1e-12);
    assert(std::abs(gmres::norm2(result.V[2]) - 1.0) < 1e-12);

    assert(std::abs(gmres::dot(result.V[0], result.V[1])) < 1e-12);
    assert(std::abs(gmres::dot(result.V[0], result.V[2])) < 1e-12);
    assert(std::abs(gmres::dot(result.V[1], result.V[2])) < 1e-12);

    // Hessenberg entries below the first subdiagonal vanish.
    assert(std::abs(result.H[2][0]) < 1e-12);

    std::println("Sequential Arnoldi test passed.");

    return 0;
}
