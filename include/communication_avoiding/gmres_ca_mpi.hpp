/**
 * @file include/communication_avoiding/gmres_ca_mpi.hpp
 * @brief Declares MPI communication-avoiding GMRES.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_GMRES_CA_MPI_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_MPI_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "communication_avoiding/ca_residual_history.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

/**
 * @brief Stores the result of an MPI CA-GMRES solve.
 */
struct CAGMRESMPIResult {
    DistributedVector x;
    CAResidualHistory residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;
};

/**
 * @brief Solves a distributed sparse linear system with CA-GMRES.
 * @param A Distributed system matrix.
 * @param b Distributed right-hand side.
 * @param x0 Distributed initial solution estimate.
 * @param config Solver configuration.
 * @return Distributed solution, residual history, iteration count, and status.
 */
CAGMRESMPIResult gmres_ca_mpi(const DistributedSparseMatrixCSR& A,
                              const DistributedVector& b,
                              const DistributedVector& x0,
                              const GMRESConfig& config);

}

#endif
