#include "communication_avoiding/gmres_ca_mpi.hpp"

#include "communication_avoiding/gmres_ca_mpi_step.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <stdexcept>

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

    while (result.iterations < config.max_iterations) {
        const Index iterations_before = result.iterations;

        CAGMRESMPICycleResult cycle =
            gmres_ca_mpi_cycle(A, b, result.x, r, beta, config);

        result.x = cycle.x;
        result.blocks_completed += cycle.blocks_completed;
        result.iterations += cycle.iterations;

        for (CAResidualSample sample : cycle.residual_history) {
            sample.iteration += iterations_before;
            result.residual_history.push_back(sample);
        }

        if (cycle.converged) {
            result.converged = true;
            return result;
        }

        r = compute_residual_mpi(A, b, result.x);
        beta = norm2_mpi(r);

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