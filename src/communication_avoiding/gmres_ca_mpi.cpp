/**
 * @file src/communication_avoiding/gmres_ca_mpi.cpp
 * @brief Implements distributed communication-avoiding GMRES.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "communication_avoiding/gmres_ca_mpi.hpp"

#include "communication_avoiding/gmres_ca_mpi_step.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace gmres {

namespace {

/**
 * @brief Computes the distributed residual vector.
 * @param A Distributed system matrix.
 * @param b Distributed right-hand side.
 * @param x Distributed solution estimate.
 * @return Distributed residual b - Ax.
 */
DistributedVector compute_residual_mpi(const DistributedSparseMatrixCSR& A,
                                       const DistributedVector& b,
                                       const DistributedVector& x)
{
    DistributedVector Ax = A.multiply(x);

    DistributedVector r = b;
    axpy_local(-1.0, Ax, r);

    return r;
}

} // namespace

CAGMRESMPIResult gmres_ca_mpi(const DistributedSparseMatrixCSR& A,
                              const DistributedVector& b,
                              const DistributedVector& x0,
                              const GMRESConfig& config)
{
    if (A.global_rows() != A.global_cols()) {
        throw std::invalid_argument("gmres_ca_mpi requires a square matrix.");
    }

    if (b.global_size() != A.global_rows()) {
        throw std::invalid_argument("gmres_ca_mpi: b has wrong global size.");
    }

    if (x0.global_size() != A.global_cols()) {
        throw std::invalid_argument("gmres_ca_mpi: x0 has wrong global size.");
    }

    if (config.restart_blocks == 0) {
        throw std::invalid_argument("gmres_ca_mpi: restart_blocks must be positive.");
    }

    if (config.s_step == 0) {
        throw std::invalid_argument("gmres_ca_mpi: s_step must be positive.");
    }

    if (config.max_iterations == 0) {
        throw std::invalid_argument("gmres_ca_mpi: max_iterations must be positive.");
    }

    check_compatible(b, x0);

    CAGMRESMPIResult result;
    result.x = x0;

    DistributedVector r = compute_residual_mpi(A, b, result.x);
    Scalar beta = norm2_mpi(r);
    const Scalar initial_beta = beta;

    result.residual_history.push_back({0, beta, true});

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    // Retained harmonic-Ritz subspace for optional recycling.
    DistributedDeflationSubspace deflation;

    // Ritz shifts for Newton-type bases, established during bootstrap.
    Vector shifts;

    // Adaptive block width carried across restart cycles.
    Index carried_s = 0;

    while (result.iterations < config.max_iterations) {
        const Index iterations_before = result.iterations;
        const Scalar beta_before = beta;

        CAGMRESMPICycleResult cycle = config.enable_recycling
            ? gmres_ca_dr_mpi_cycle(A, b, result.x, r, beta, initial_beta, config,
                                    deflation, shifts, carried_s)
            : gmres_ca_mpi_cycle(A, b, result.x, r, beta, initial_beta, config,
                                 shifts, carried_s);

        carried_s = cycle.adapted_s;

        result.x = cycle.x;
        result.blocks_completed += cycle.blocks_completed;
        result.iterations += cycle.iterations;

        for (CAResidualSample sample : cycle.residual_history) {
            sample.iteration += iterations_before;
            result.residual_history.push_back(sample);
        }

        if (shifts.empty() && !cycle.bootstrap_shifts.empty()) {
            shifts = std::move(cycle.bootstrap_shifts);
        }

        // Recompute the true residual before convergence testing.
        r = compute_residual_mpi(A, b, result.x);
        beta = norm2_mpi(r);

        if (config.enable_recycling) {
            // Discard a recycle subspace when its cycle did not make progress.
            const bool made_progress = std::isfinite(beta) && beta < beta_before;
            deflation = made_progress ? std::move(cycle.next_deflation)
                                      : DistributedDeflationSubspace();
        }

        result.residual_history.push_back({result.iterations, beta, true});

        if (beta < config.tolerance * initial_beta) {
            result.converged = true;
            return result;
        }

        if (cycle.iterations == 0) {
            break;
        }
    }

    result.converged = false;
    return result;
}

} // namespace gmres
