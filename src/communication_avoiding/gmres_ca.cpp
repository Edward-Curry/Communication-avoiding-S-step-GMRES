#include "communication_avoiding/gmres_ca.hpp"

#include "common/vector_ops.hpp"
#include "communication_avoiding/gmres_ca_step.hpp"

#include <stdexcept>

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

    result.residual_history.push_back(beta);

    if (beta < config.tolerance) {
        result.converged = true;
        return result;
    }

    while (result.iterations < config.max_iterations) {
        CAGMRESCycleResult cycle =
            gmres_ca_cycle(A, b, result.x, r, beta, config);

        result.x = cycle.x;
        result.blocks_completed += cycle.blocks_completed;
        result.iterations += cycle.iterations;

        for (Scalar residual : cycle.residual_history) {
            result.residual_history.push_back(residual);
        }

        if (cycle.converged) {
            result.converged = true;
            return result;
        }

        r = compute_residual(A, b, result.x);
        beta = norm2(r);

        result.residual_history.push_back(beta);

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