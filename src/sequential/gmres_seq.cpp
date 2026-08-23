/**
 * @file src/sequential/gmres_seq.cpp
 * @brief Implements sequential restarted GMRES.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "sequential/gmres_seq.hpp"

#include "common/vector_ops.hpp"
#include "sequential/gmres_seq_step.hpp"

#include <print>
#include <stdexcept>

namespace gmres
{
    namespace
    {
        /**
         * @brief Computes the residual vector for a sequential system.
         * @param A System matrix.
         * @param b Right-hand side.
         * @param x Current solution estimate.
         * @return Residual vector b - Ax.
         */
        Vector compute_residual(const SparseMatrixCSR& A,
                                const Vector& b,
                                const Vector& x)
        {
            Vector Ax = A.multiply(x);
            Vector r = b;

            axpy(-1.0, Ax, r);

            return r;
        }
    }

    GMRESResult gmres_seq(const SparseMatrixCSR& A,
                          const Vector& b,
                          const Vector& x0,
                          const GMRESConfig& config)
    {
        if (A.rows() != A.cols())
        {
            throw std::invalid_argument("GMRES requires a square matrix.");
        }

        if (b.size() != A.rows())
        {
            throw std::invalid_argument("Right-hand side size must match matrix rows.");
        }

        if (x0.size() != A.cols())
        {
            throw std::invalid_argument("Initial guess size must match matrix columns.");
        }

        if (config.restart == 0)
        {
            throw std::invalid_argument("GMRES restart value must be greater than zero.");
        }

        if (config.max_iterations == 0)
        {
            throw std::invalid_argument("GMRES max_iterations must be greater than zero.");
        }

        GMRESResult result;
        result.x = x0;

        Vector r = compute_residual(A, b, result.x);
        Scalar beta = norm2(r);
        const Scalar initial_beta = beta;

        result.residual_history.push_back(beta);

        if (config.verbose)
        {
            std::println("GMRES initial residual = {}", beta);
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

            GMRESCycleResult cycle = gmres_seq_cycle(A,
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

            r = compute_residual(A, b, result.x);
            beta = norm2(r);

            result.residual_history.push_back(beta);

            if (config.verbose)
            {
                std::println("GMRES restart residual = {}", beta);
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
