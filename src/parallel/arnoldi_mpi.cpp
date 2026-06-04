#include "parallel/arnoldi_mpi.hpp"

#include "parallel/distributed_orthogonalization.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <stdexcept>

namespace gmres
{
    ArnoldiMPIResult arnoldi_mpi(const DistributedSparseMatrixCSR& A,
                                 const DistributedVector& r0,
                                 Index m)
    {
        if (r0.global_size() != A.global_rows())
        {
            throw std::invalid_argument("Initial residual size must match matrix rows.");
        }

        if (r0.communicator() != A.communicator())
        {
            throw std::invalid_argument("Matrix and residual must use the same MPI communicator.");
        }

        ArnoldiMPIResult result;

        result.beta = norm2_mpi(r0);

        if (result.beta == 0.0)
        {
            throw std::runtime_error("MPI Arnoldi cannot start from a zero residual.");
        }

        result.V.reserve(m + 1);

        DistributedVector v0 = r0;
        scal_local(1.0 / result.beta, v0);
        result.V.push_back(v0);

        result.H.assign(m + 1, Vector(m, 0.0));

        for (Index j = 0; j < m; ++j)
        {
            DistributedVector w = A.multiply(result.V[j]);

            Vector h_column;
            modified_gram_schmidt_mpi(result.V, w, h_column);

            for (Index i = 0; i < h_column.size(); ++i)
            {
                result.H[i][j] = h_column[i];
            }

            result.V.push_back(w);
        }

        return result;
    }
}