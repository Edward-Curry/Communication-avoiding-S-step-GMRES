#include "parallel/gmres_mpi_step.hpp"

#include "common/givens.hpp"
#include "parallel/distributed_orthogonalization.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <cmath>
#include <print>
#include <stdexcept>

namespace gmres
{
    namespace
    {
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
                    throw std::runtime_error("Zero diagonal entry in MPI GMRES back substitution.");
                }

                y[i] = sum / H[i][i];
            }

            return y;
        }

        void update_solution_mpi(DistributedVector& x,
                                 const DistributedVectorList& V,
                                 const Vector& y)
        {
            for (Index i = 0; i < y.size(); ++i)
            {
                axpy_local(y[i], V[i], x);
            }
        }
    }

    GMRESMPICycleResult gmres_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                        const DistributedVector& b,
                                        const DistributedVector& x_start,
                                        const DistributedVector& r_start,
                                        Scalar beta,
                                        const GMRESConfig& config)
    {
        if (b.global_size() != A.global_rows())
        {
            throw std::invalid_argument("Right-hand side size must match matrix rows.");
        }

        if (x_start.global_size() != A.global_cols())
        {
            throw std::invalid_argument("Starting solution size must match matrix columns.");
        }

        if (r_start.global_size() != A.global_rows())
        {
            throw std::invalid_argument("Starting residual size must match matrix rows.");
        }

        if (b.communicator() != A.communicator() ||
            x_start.communicator() != A.communicator() ||
            r_start.communicator() != A.communicator())
        {
            throw std::invalid_argument("Matrix and vectors must use the same MPI communicator.");
        }

        if (config.restart == 0)
        {
            throw std::invalid_argument("MPI GMRES restart value must be greater than zero.");
        }

        GMRESMPICycleResult result;
        result.x = x_start;

        if (beta == 0.0)
        {
            result.converged = true;
            return result;
        }

        DistributedVectorList V;
        V.reserve(config.restart + 1);

        DistributedVector v0 = r_start;
        scal_local(1.0 / beta, v0);
        V.push_back(v0);

        DenseMatrix H(config.restart + 1, Vector(config.restart, 0.0));

        Vector g(config.restart + 1, 0.0);
        g[0] = beta;

        Vector cosines(config.restart, 0.0);
        Vector sines(config.restart, 0.0);

        Index inner_iterations = 0;

        for (Index j = 0; j < config.restart; ++j)
        {
            DistributedVector w = A.multiply(V[j]);

            Vector h_column;
            modified_gram_schmidt_mpi(V, w, h_column);

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
                std::println("MPI GMRES inner iteration {} residual = {}",
                             result.iterations,
                             residual_norm);
            }

            if (residual_norm < config.tolerance)
            {
                Vector y = solve_upper_triangular(H, g, inner_iterations);
                update_solution_mpi(result.x, V, y);

                result.converged = true;
                return result;
            }
        }

        Vector y = solve_upper_triangular(H, g, inner_iterations);
        update_solution_mpi(result.x, V, y);

        result.converged = false;
        return result;
    }
}
