/**
 * @file include/sequential/gmres_seq.hpp
 * @brief Declares sequential restarted GMRES.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef SEQUENTIAL_GMRES_SEQ_HPP
#define SEQUENTIAL_GMRES_SEQ_HPP

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

namespace gmres
{
    /**
     * @brief Stores the result of a sequential GMRES solve.
     */
    struct GMRESResult
    {
        Vector x;
        Vector residual_history;
        Index iterations = 0;
        bool converged = false;
    };

    /**
     * @brief Solves a sparse linear system with restarted GMRES.
     * @param A System matrix.
     * @param b Right-hand side.
     * @param x0 Initial solution estimate.
     * @param config Solver configuration.
     * @return Final solution, residual history, iteration count, and status.
     */
    GMRESResult gmres_seq(const SparseMatrixCSR& A,
                          const Vector& b,
                          const Vector& x0,
                          const GMRESConfig& config);
}

#endif
