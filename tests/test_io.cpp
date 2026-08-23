/**
 * @file tests/test_io.cpp
 * @brief Tests Matrix Market input and CSV output.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "common/io.hpp"
#include "common/sparse_matrix.hpp"

#include <cassert>
#include <cmath>
#include <print>

/**
 * @brief Runs the input and output test.
 * @return Zero when all assertions pass.
 */
int main()
{
    gmres::SparseMatrixCSR A =
        gmres::read_matrix_market("data/matrices/test_3x3.mtx");

    assert(A.rows() == 3);
    assert(A.cols() == 3);
    assert(A.nonzeros() == 5);

    gmres::Vector x = {1.0, 2.0, 3.0};
    gmres::Vector y = A.multiply(x);

    assert(y.size() == 3);
    assert(std::abs(y[0] - 2.0) < 1e-12);
    assert(std::abs(y[1] - 18.0) < 1e-12);
    assert(std::abs(y[2] - 23.0) < 1e-12);

    gmres::SparseMatrixCSR symmetric =
        gmres::read_matrix_market(
            "data/matrices/test_symmetric_3x3.mtx");

    assert(symmetric.rows() == 3);
    assert(symmetric.cols() == 3);
    assert(symmetric.nonzeros() == 5);

    gmres::Vector symmetric_y = symmetric.multiply(x);

    assert(symmetric_y.size() == 3);
    assert(std::abs(symmetric_y[0] - 10.0) < 1e-12);
    assert(std::abs(symmetric_y[1] - 10.0) < 1e-12);
    assert(std::abs(symmetric_y[2] - 18.0) < 1e-12);

    gmres::Vector residuals = {1.0, 0.5, 0.25, 0.125};
    gmres::write_residual_history("data/outputs/test_residuals.csv", residuals);

    gmres::write_vector_csv("data/outputs/test_solution.csv", y);

    std::println("IO test passed.");

    return 0;
}
