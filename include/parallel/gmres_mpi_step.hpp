/**
 * @file include/parallel/gmres_mpi_step.hpp
 * @brief Declares one MPI GMRES restart cycle.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_GMRES_MPI_STEP_HPP
#define PARALLEL_GMRES_MPI_STEP_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    /**
     * @brief Stores the result of one MPI GMRES cycle.
     */
    struct GMRESMPICycleResult
    {
        DistributedVector x;
        Vector residual_history;
        Index iterations = 0;
        bool converged = false;
    };

    /**
     * @brief Executes one distributed restarted GMRES cycle.
     * @param A Distributed system matrix.
     * @param b Distributed right-hand side.
     * @param x_start Distributed solution at cycle entry.
     * @param r_start Distributed residual at cycle entry.
     * @param beta Global norm of r_start.
     * @param initial_beta Initial global residual norm.
     * @param config Solver configuration.
     * @return Updated solution, residual history, iteration count, and status.
     */
    GMRESMPICycleResult gmres_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                        const DistributedVector& b,
                                        const DistributedVector& x_start,
                                        const DistributedVector& r_start,
                                        Scalar beta,
                                        Scalar initial_beta,
                                        const GMRESConfig& config);
}

#endif
