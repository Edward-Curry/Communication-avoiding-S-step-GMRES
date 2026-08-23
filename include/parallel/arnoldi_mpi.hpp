/**
 * @file include/parallel/arnoldi_mpi.hpp
 * @brief Declares MPI Arnoldi iteration.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_ARNOLDI_MPI_HPP
#define PARALLEL_ARNOLDI_MPI_HPP

#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    /**
     * @brief Stores a distributed Arnoldi basis and Hessenberg relation.
     */
    struct ArnoldiMPIResult
    {
        DistributedVectorList V;
        DenseMatrix H;
        Scalar beta = 0.0;
    };

    /**
     * @brief Generates a distributed m-step Arnoldi factorisation.
     * @param A Distributed system matrix.
     * @param r0 Distributed initial residual.
     * @param m Number of Arnoldi steps.
     * @return Distributed basis, replicated Hessenberg matrix, and norm.
     */
    ArnoldiMPIResult arnoldi_mpi(const DistributedSparseMatrixCSR& A,
                                 const DistributedVector& r0,
                                 Index m);
}

#endif
