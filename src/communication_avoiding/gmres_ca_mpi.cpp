#include "communication_avoiding/gmres_ca_mpi.hpp"

#include "communication_avoiding/gmres_ca_mpi_step.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace gmres {

namespace {

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

    result.residual_history.push_back({0, beta, true});

    if (beta < config.tolerance) {
        result.converged = true;
        return result;
    }

    // GMRES-DR deflation subspace carried across restart cycles. Starts empty
    // (k == 0), so the first cycle runs undeflated. Only used when
    // config.enable_recycling is set.
    DistributedDeflationSubspace deflation;

    // Newton/ScaledNewton shifts, computed once (from the first cycle's
    // Monomial-bootstrap Hessenberg) and reused for the rest of the solve.
    Vector shifts;

    while (result.iterations < config.max_iterations) {
        const Index iterations_before = result.iterations;
        const Scalar beta_before = beta;

        CAGMRESMPICycleResult cycle = config.enable_recycling
            ? gmres_ca_dr_mpi_cycle(A, b, result.x, r, beta, config, deflation, shifts)
            : gmres_ca_mpi_cycle(A, b, result.x, r, beta, config, shifts);

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

        // Always verify against a freshly recomputed residual rather than
        // trusting the cycle's internal least-squares estimate outright: over
        // many blocks - especially across a deflated restart - the estimate can
        // drift from the true residual, and this recompute is the safety net
        // that catches it before reporting false convergence.
        r = compute_residual_mpi(A, b, result.x);
        beta = norm2_mpi(r);

        if (config.enable_recycling) {
            // Adopt this cycle's deflation subspace, unless it failed to reduce
            // the residual: a stalled or non-finite deflated restart is
            // discarded so the next cycle falls back to plain GMRES (self-
            // healing against a weak subspace on hard/non-symmetric systems).
            const bool made_progress = std::isfinite(beta) && beta < beta_before;
            deflation = made_progress ? std::move(cycle.next_deflation)
                                      : DistributedDeflationSubspace();
        }

        result.residual_history.push_back({result.iterations, beta, true});

        if (beta < config.tolerance) {
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