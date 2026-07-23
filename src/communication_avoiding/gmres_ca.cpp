#include "communication_avoiding/gmres_ca.hpp"

#include "common/dense_block.hpp"
#include "common/vector_ops.hpp"
#include "communication_avoiding/gmres_ca_step.hpp"

#include <stdexcept>
#include <utility>

namespace gmres {

namespace {

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

    result.residual_history.push_back({0, beta, true});

    if (beta < config.tolerance) {
        result.converged = true;
        return result;
    }

    // Recycled subspace carried across restart cycles ("keep most recent
    // winner": each cycle's own best blocks, when there are any, replace this
    // outright). Starts empty, so the first cycle is unaffected.
    DenseBlock recycled_U;

    // Newton/ScaledNewton shifts, computed once (from the first cycle's
    // Monomial-bootstrap Hessenberg) and reused for the rest of the solve.
    // Starts empty, so the first cycle always runs in bootstrap mode when
    // config.polynomial_basis wants Newton/ScaledNewton.
    Vector shifts;

    while (result.iterations < config.max_iterations) {
        const Index iterations_before = result.iterations;

        CAGMRESCycleResult cycle =
            gmres_ca_cycle(A, b, result.x, r, beta, config, recycled_U, shifts);

        result.x = cycle.x;
        result.blocks_completed += cycle.blocks_completed;
        result.iterations += cycle.iterations;

        for (CAResidualSample sample : cycle.residual_history) {
            sample.iteration += iterations_before;
            result.residual_history.push_back(sample);
        }

        if (config.enable_recycling && cycle.recycle_candidate_block.cols() > 0) {
            recycled_U = std::move(cycle.recycle_candidate_block);
        }

        if (shifts.empty() && !cycle.bootstrap_shifts.empty()) {
            shifts = std::move(cycle.bootstrap_shifts);
        }

        // Always verify against a freshly recomputed residual rather than
        // trusting the cycle's internal (Givens-based) estimate outright:
        // over many rotations - especially across a recycled-subspace seed -
        // the estimate can drift from the true residual, and this recompute
        // is the safety net that catches it before reporting false
        // convergence.
        r = compute_residual(A, b, result.x);
        beta = norm2(r);

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