/**
 * @file src/sequential/arnoldi_seq.cpp
 * @brief Implements sequential Arnoldi factorization.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "sequential/arnoldi_seq.hpp"

#include "common/orthogonalization.hpp"
#include "common/vector_ops.hpp"

#include <stdexcept>

namespace gmres
{
    ArnoldiResult arnoldi_seq(const SparseMatrixCSR& A,
                              const Vector& r0,
                              Index m)
    {
        if (r0.size() != A.rows())
        {
            throw std::invalid_argument("Initial residual size must match matrix rows.");
        }

        ArnoldiResult result;

        result.beta = norm2(r0);

        if (result.beta == 0.0)
        {
            throw std::runtime_error("Arnoldi cannot start from a zero residual.");
        }

        result.V.reserve(m + 1);

        Vector v0 = r0;
        scal(1.0 / result.beta, v0);
        result.V.push_back(v0);

        result.H.assign(m + 1, Vector(m, 0.0));

        for (Index j = 0; j < m; ++j)
        {
            Vector w = A.multiply(result.V[j]);

            Vector h_column;
            modified_gram_schmidt(result.V, w, h_column);

            for (Index i = 0; i < h_column.size(); ++i)
            {
                result.H[i][j] = h_column[i];
            }

            result.V.push_back(w);
        }

        return result;
    }
}
