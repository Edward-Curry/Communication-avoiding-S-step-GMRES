/**
 * @file src/communication_avoiding/gmres_ca.cpp
 * @brief Implements sequential communication-avoiding GMRES.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "communication_avoiding/gmres_ca.hpp"

#include "common/dense_block.hpp"
#include "common/vector_ops.hpp"
#include "communication_avoiding/gmres_ca_step.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace gmres {

namespace {

/**
 * @brief Computes the residual vector for a sequential system.
 * @param A System matrix.
 * @param b Right-hand side.
 * @param x Current solution estimate.
 * @return Residual vector b - Ax.
 */
Vector compute_residual(const SparseMatrixCSR& A, const Vector& b, const Vector& x)
{
    Vector Ax = A.multiply(x);

    Vector r = b;
    axpy(-1.0, Ax, r);

    return r;
}

} // namespace

CAGMRESResult gmres_ca(const SparseMatrixCSR& A,
                       const Vector& b,
                       const Vector& x0,
                       const GMRESConfig& config)
{
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("gmres_ca requires a square matrix.");
    }

    if (b.size() != A.rows()) {
        throw std::invalid_argument("gmres_ca: b has wrong size.");
    }

    if (x0.size() != A.cols()) {
        throw std::invalid_argument("gmres_ca: x0 has wrong size.");
    }

    if (config.restart_blocks == 0) {
        throw std::invalid_argument("gmres_ca: restart_blocks must be positive.");
    }

    if (config.s_step == 0) {
        throw std::invalid_argument("gmres_ca: s_step must be positive.");
    }

    if (config.max_iterations == 0) {
        throw std::invalid_argument("gmres_ca: max_iterations must be positive.");
    }

    CAGMRESResult result;
    result.x = x0;

    Vector r = compute_residual(A, b, result.x);
    Scalar beta = norm2(r);
    const Scalar initial_beta = beta;

    result.residual_history.push_back({0, beta, true});

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    // Retained harmonic-Ritz subspace for optional recycling.
    DeflationSubspace deflation;

    // Ritz shifts for Newton-type bases, established during bootstrap.
    Vector shifts;

    // Adaptive block width carried across restart cycles.
    Index carried_s = 0;

    while (result.iterations < config.max_iterations) {
        const Index iterations_before = result.iterations;
        const Scalar beta_before = beta;

        CAGMRESCycleResult cycle = config.enable_recycling
            ? gmres_ca_dr_cycle(A, b, result.x, r, beta, initial_beta, config,
                                deflation, shifts, carried_s)
            : gmres_ca_cycle(A, b, result.x, r, beta, initial_beta, config,
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
        r = compute_residual(A, b, result.x);
        beta = norm2(r);

        if (config.enable_recycling) {
            // Discard a recycle subspace when its cycle did not make progress.
            const bool made_progress = std::isfinite(beta) && beta < beta_before;
            deflation = made_progress ? std::move(cycle.next_deflation)
                                      : DeflationSubspace();
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
