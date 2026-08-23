/**
 * @file include/communication_avoiding/gmres_ca_mpi_step.hpp
 * @brief Declares MPI CA-GMRES restart cycles.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_GMRES_CA_MPI_STEP_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_MPI_STEP_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "communication_avoiding/ca_residual_history.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

/**
 * @brief Carries a distributed GMRES-DR subspace between cycles.
 */
struct DistributedDeflationSubspace {
    /// @brief Row-distributed orthonormal seed vectors.
    DistributedDenseBlock V;
    /// @brief Replicated Arnoldi relation for the retained subspace.
    DenseMatrix Hbar;
    /// @brief Number of retained harmonic Ritz vectors.
    Index k = 0;
};

/**
 * @brief Stores the result of one MPI CA-GMRES cycle.
 */
struct CAGMRESMPICycleResult {
    DistributedVector x;
    CAResidualHistory residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;

    /// @brief Deflation subspace prepared for the next cycle.
    DistributedDeflationSubspace next_deflation;

    /// @brief Replicated Leja-ordered Ritz shifts from a bootstrap cycle.
    Vector bootstrap_shifts;

    /// @brief Width to carry into the next adaptive cycle.
    Index adapted_s = 0;
};

/**
 * @brief Executes one non-deflated MPI CA-GMRES restart cycle.
 * @param A Distributed system matrix.
 * @param b Distributed right-hand side.
 * @param x_start Distributed solution at cycle entry.
 * @param r_start Distributed residual at cycle entry.
 * @param beta Global norm of r_start.
 * @param initial_beta Initial global residual norm.
 * @param config Solver configuration.
 * @param shifts Replicated Newton shifts, or an empty vector for bootstrap.
 * @param carried_s Width accepted by the preceding cycle.
 * @return Updated solution, residual history, and adaptive state.
 */
CAGMRESMPICycleResult gmres_ca_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                         const DistributedVector& b,
                                         const DistributedVector& x_start,
                                         const DistributedVector& r_start,
                                         Scalar beta,
                                         Scalar initial_beta,
                                         const GMRESConfig& config,
                                         const Vector& shifts = Vector(),
                                         Index carried_s = 0);

/**
 * @brief Executes one deflated MPI CA-GMRES restart cycle.
 * @param A Distributed system matrix.
 * @param b Distributed right-hand side.
 * @param x_start Distributed solution at cycle entry.
 * @param r_start Distributed residual at cycle entry.
 * @param beta Global norm of r_start.
 * @param initial_beta Initial global residual norm.
 * @param config Solver configuration.
 * @param deflation Retained distributed subspace from the preceding cycle.
 * @param shifts Replicated Newton shifts, or an empty vector for bootstrap.
 * @param carried_s Width accepted by the preceding cycle.
 * @return Updated solution, residual history, and next deflation subspace.
 */
CAGMRESMPICycleResult gmres_ca_dr_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                            const DistributedVector& b,
                                            const DistributedVector& x_start,
                                            const DistributedVector& r_start,
                                            Scalar beta,
                                            Scalar initial_beta,
                                            const GMRESConfig& config,
                                            const DistributedDeflationSubspace& deflation =
                                                DistributedDeflationSubspace(),
                                            const Vector& shifts = Vector(),
                                            Index carried_s = 0);

}

#endif
