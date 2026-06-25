#ifndef PARALLEL_DISTRIBUTED_UTILS_HPP
#define PARALLEL_DISTRIBUTED_UTILS_HPP

#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    DistributedVector compute_residual_mpi(const DistributedSparseMatrixCSR& A,
                                           const DistributedVector& b,
                                           const DistributedVector& x);

    Scalar residual_norm_mpi(const DistributedSparseMatrixCSR& A,
                             const DistributedVector& b,
                             const DistributedVector& x);

    Scalar relative_residual_norm_mpi(const DistributedSparseMatrixCSR& A,
                                      const DistributedVector& b,
                                      const DistributedVector& x);

    Scalar forward_error_norm_mpi(const DistributedVector& x,
                                  const DistributedVector& x_true);

    Scalar relative_forward_error_mpi(const DistributedVector& x,
                                      const DistributedVector& x_true);

    bool has_converged_absolute_residual_mpi(Scalar residual,
                                             Scalar tolerance);

    bool has_converged_relative_residual_mpi(Scalar relative_residual,
                                             Scalar tolerance);
}

#endif
