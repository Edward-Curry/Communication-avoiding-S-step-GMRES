/**
 * @file src/parallel/gmres_mpi.cpp
 * @brief Implements distributed restarted GMRES.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "parallel/gmres_mpi.hpp"

#include "parallel/distributed_vector_ops.hpp"
#include "parallel/gmres_mpi_step.hpp"

#include <print>
#include <stdexcept>

namespace gmres
{
    namespace
    {
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
    }

    GMRESMPIResult gmres_mpi(const DistributedSparseMatrixCSR& A,
                             const DistributedVector& b,
                             const DistributedVector& x0,
                             const GMRESConfig& config)
    {
        if (A.global_rows() != A.global_cols())
        {
            throw std::invalid_argument("MPI GMRES requires a square matrix.");
        }

        if (b.global_size() != A.global_rows())
        {
            throw std::invalid_argument("Right-hand side size must match matrix rows.");
        }

        if (x0.global_size() != A.global_cols())
        {
            throw std::invalid_argument("Initial guess size must match matrix columns.");
        }

        if (b.communicator() != A.communicator() ||
            x0.communicator() != A.communicator())
        {
            throw std::invalid_argument("Matrix and vectors must use the same MPI communicator.");
        }

        if (config.restart == 0)
        {
            throw std::invalid_argument("MPI GMRES restart value must be greater than zero.");
        }

        if (config.max_iterations == 0)
        {
            throw std::invalid_argument("MPI GMRES max_iterations must be greater than zero.");
        }

        GMRESMPIResult result;
        result.x = x0;

        DistributedVector r = compute_residual_mpi(A, b, result.x);
        Scalar beta = norm2_mpi(r);
        const Scalar initial_beta = beta;

        result.residual_history.push_back(beta);

        if (config.verbose)
        {
            std::println("MPI GMRES initial residual = {}", beta);
        }

        if (beta == 0.0)
        {
            result.converged = true;
            return result;
        }

        while (result.iterations < config.max_iterations)
        {
            GMRESConfig cycle_config = config;

            Index remaining_iterations = config.max_iterations - result.iterations;

            if (cycle_config.restart > remaining_iterations)
            {
                cycle_config.restart = remaining_iterations;
            }

            GMRESMPICycleResult cycle = gmres_mpi_cycle(A,
                                                        b,
                                                        result.x,
                                                        r,
                                                        beta,
                                                        initial_beta,
                                                        cycle_config);

            result.x = cycle.x;
            result.iterations += cycle.iterations;

            for (Scalar residual : cycle.residual_history)
            {
                result.residual_history.push_back(residual);
            }

            if (cycle.converged)
            {
                result.converged = true;
                return result;
            }

            r = compute_residual_mpi(A, b, result.x);
            beta = norm2_mpi(r);

            result.residual_history.push_back(beta);

            if (config.verbose)
            {
                std::println("MPI GMRES restart residual = {}", beta);
            }

            if (beta < config.tolerance * initial_beta)
            {
                result.converged = true;
                return result;
            }
        }

        result.converged = false;
        return result;
    }
}
