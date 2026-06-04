#ifndef PARALLEL_ARNOLDI_MPI_HPP
#define PARALLEL_ARNOLDI_MPI_HPP

#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    struct ArnoldiMPIResult
    {
        DistributedVectorList V;
        DenseMatrix H;
        Scalar beta = 0.0;
    };

    ArnoldiMPIResult arnoldi_mpi(const DistributedSparseMatrixCSR& A,
                                 const DistributedVector& r0,
                                 Index m);
}

#endif