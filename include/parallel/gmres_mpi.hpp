/**
 * @file include/parallel/gmres_mpi.hpp
 * @brief Declares MPI restarted GMRES.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_GMRES_MPI_HPP
#define PARALLEL_GMRES_MPI_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    /**
     * @brief Stores the result of an MPI GMRES solve.
     */
    struct GMRESMPIResult
    {
        DistributedVector x;
        Vector residual_history;
        Index iterations = 0;
        bool converged = false;
    };

    /**
     * @brief Solves a distributed sparse linear system with restarted GMRES.
     * @param A Distributed system matrix.
     * @param b Distributed right-hand side.
     * @param x0 Distributed initial solution estimate.
     * @param config Solver configuration.
     * @return Distributed solution, residual history, iteration count, and status.
     */
    GMRESMPIResult gmres_mpi(const DistributedSparseMatrixCSR& A,
                             const DistributedVector& b,
                             const DistributedVector& x0,
                             const GMRESConfig& config);
}

#endif
