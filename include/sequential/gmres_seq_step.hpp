/**
 * @file include/sequential/gmres_seq_step.hpp
 * @brief Declares one sequential GMRES restart cycle.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef SEQUENTIAL_GMRES_SEQ_STEP_HPP
#define SEQUENTIAL_GMRES_SEQ_STEP_HPP

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

namespace gmres
{
    /**
     * @brief Stores the result of one sequential GMRES cycle.
     */
    struct GMRESCycleResult
    {
        Vector x;
        Vector residual_history;
        Index iterations = 0;
        bool converged = false;
    };

    /**
     * @brief Executes one restarted GMRES cycle.
     * @param A System matrix.
     * @param b Right-hand side.
     * @param x_start Solution at cycle entry.
     * @param r_start Residual at cycle entry.
     * @param beta Norm of r_start.
     * @param initial_beta Initial residual norm for relative convergence.
     * @param config Solver configuration.
     * @return Updated solution, residual history, iteration count, and status.
     */
    GMRESCycleResult gmres_seq_cycle(const SparseMatrixCSR& A,
                                     const Vector& b,
                                     const Vector& x_start,
                                     const Vector& r_start,
                                     Scalar beta,
                                     Scalar initial_beta,
                                     const GMRESConfig& config);
}

#endif
