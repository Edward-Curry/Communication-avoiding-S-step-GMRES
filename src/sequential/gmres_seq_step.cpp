/**
 * @file src/sequential/gmres_seq_step.cpp
 * @brief Implements one sequential restarted GMRES cycle.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "sequential/gmres_seq_step.hpp"

#include "common/givens.hpp"
#include "common/orthogonalization.hpp"
#include "common/vector_ops.hpp"

#include <cmath>
#include <print>
#include <stdexcept>

namespace gmres
{
    namespace
    {
        /**
         * @brief Solves the leading upper-triangular GMRES system.
         * @param H Rotated Hessenberg matrix.
         * @param g Rotated residual right-hand side.
         * @param k Dimension of the system.
         * @return Coefficients of the GMRES correction.
         */
        Vector solve_upper_triangular(const DenseMatrix& H,
                                      const Vector& g,
                                      Index k)
        {
            Vector y(k, 0.0);

            for (Index offset = 0; offset < k; ++offset)
            {
                Index i = k - 1 - offset;

                Scalar sum = g[i];

                for (Index j = i + 1; j < k; ++j)
                {
                    sum -= H[i][j] * y[j];
                }

                if (H[i][i] == 0.0)
                {
                    throw std::runtime_error("Zero diagonal entry in GMRES back substitution.");
                }

                y[i] = sum / H[i][i];
            }

            return y;
        }

        /**
         * @brief Applies a Krylov-basis correction to the solution.
         * @param x Solution estimate updated in place.
         * @param V Krylov basis vectors.
         * @param y Correction coefficients.
         */
        void update_solution(Vector& x,
                             const VectorList& V,
                             const Vector& y)
        {
            for (Index i = 0; i < y.size(); ++i)
            {
                axpy(y[i], V[i], x);
            }
        }
    }

    GMRESCycleResult gmres_seq_cycle(const SparseMatrixCSR& A,
                                     const Vector& b,
                                     const Vector& x_start,
                                     const Vector& r_start,
                                     Scalar beta,
                                     Scalar initial_beta,
                                     const GMRESConfig& config)
    {
        if (b.size() != A.rows())
        {
            throw std::invalid_argument("Right-hand side size must match matrix rows.");
        }

        if (x_start.size() != A.cols())
        {
            throw std::invalid_argument("Starting solution size must match matrix columns.");
        }

        if (r_start.size() != A.rows())
        {
            throw std::invalid_argument("Starting residual size must match matrix rows.");
        }

        if (config.restart == 0)
        {
            throw std::invalid_argument("GMRES restart value must be greater than zero.");
        }

        if (beta == 0.0)
        {
            GMRESCycleResult result;
            result.x = x_start;
            result.converged = true;
            return result;
        }

        GMRESCycleResult result;
        result.x = x_start;

        VectorList V;
        V.reserve(config.restart + 1);

        Vector v0 = r_start;
        scal(1.0 / beta, v0);
        V.push_back(v0);

        DenseMatrix H(config.restart + 1, Vector(config.restart, 0.0));

        Vector g(config.restart + 1, 0.0);
        g[0] = beta;

        Vector cosines(config.restart, 0.0);
        Vector sines(config.restart, 0.0);

        Index inner_iterations = 0;

        for (Index j = 0; j < config.restart; ++j)
        {
            Vector w = A.multiply(V[j]);

            Vector h_column;
            modified_gram_schmidt(V, w, h_column);

            for (Index i = 0; i < h_column.size(); ++i)
            {
                H[i][j] = h_column[i];
            }

            V.push_back(w);

            for (Index i = 0; i < j; ++i)
            {
                apply_givens(cosines[i],
                              sines[i],
                              H[i][j],
                              H[i + 1][j]);
            }

            generate_givens(H[j][j],
                             H[j + 1][j],
                             cosines[j],
                             sines[j]);

            apply_givens(cosines[j],
                          sines[j],
                          H[j][j],
                          H[j + 1][j]);

            apply_givens(cosines[j],
                          sines[j],
                          g[j],
                          g[j + 1]);

            Scalar residual_norm = std::abs(g[j + 1]);

            result.residual_history.push_back(residual_norm);
            ++result.iterations;
            inner_iterations = j + 1;

            if (config.verbose)
            {
                std::println("GMRES inner iteration {} residual = {}",
                             result.iterations,
                             residual_norm);
            }

            if (residual_norm < config.tolerance * initial_beta)
            {
                Vector y = solve_upper_triangular(H, g, inner_iterations);
                update_solution(result.x, V, y);

                result.converged = true;
                return result;
            }
        }

        Vector y = solve_upper_triangular(H, g, inner_iterations);
        update_solution(result.x, V, y);

        result.converged = false;
        return result;
    }
}
