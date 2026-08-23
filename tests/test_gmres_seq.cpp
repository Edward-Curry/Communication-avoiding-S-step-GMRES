/**
 * @file tests/test_gmres_seq.cpp
 * @brief Tests sequential restarted GMRES.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "sequential/gmres_seq.hpp"

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/vector_ops.hpp"

#include <cassert>
#include <cmath>
#include <print>
#include <vector>

/**
 * @brief Runs the sequential GMRES test.
 * @return Zero when all assertions pass.
 */
int main()
{
    // Two-by-two reference matrix in CSR format.
    gmres::Vector values = {
        4.0, 1.0,
        1.0, 3.0
    };

    std::vector<gmres::Index> col_indices = {
        0, 1,
        0, 1
    };

    std::vector<gmres::Index> row_ptr = {
        0, 2, 4
    };

    gmres::SparseMatrixCSR A(2, 2, values, col_indices, row_ptr);

    // The right-hand side has exact solution [1, 2].
    gmres::Vector b = {6.0, 7.0};

    gmres::Vector x0 = {0.0, 0.0};

    gmres::GMRESConfig config;
    config.restart = 2;
    config.max_iterations = 10;
    config.tolerance = 1e-10;
    config.verbose = true;

    gmres::GMRESResult result = gmres::gmres_seq(A, b, x0, config);

    assert(result.converged);
    assert(result.iterations <= config.max_iterations);

    assert(result.x.size() == 2);
    assert(std::abs(result.x[0] - 1.0) < 1e-8);
    assert(std::abs(result.x[1] - 2.0) < 1e-8);

    gmres::Vector Ax = A.multiply(result.x);
    gmres::Vector residual = b;
    gmres::axpy(-1.0, Ax, residual);

    assert(gmres::norm2(residual) < config.tolerance);

    std::println("Sequential GMRES test passed.");

    return 0;

}
