#include "parallel/distributed_utils.hpp"

#include "parallel/distributed_vector_ops.hpp"

namespace gmres
{
    DistributedVector compute_residual_mpi(const DistributedSparseMatrixCSR& A,
                                           const DistributedVector& b,
                                           const DistributedVector& x)
    {
        DistributedVector residual = b;
        DistributedVector Ax = A.multiply(x);

        axpy_local(-1.0, Ax, residual);

        return residual;
    }

    Scalar residual_norm_mpi(const DistributedSparseMatrixCSR& A,
                             const DistributedVector& b,
                             const DistributedVector& x)
    {
        return norm2_mpi(compute_residual_mpi(A, b, x));
    }

    Scalar relative_residual_norm_mpi(const DistributedSparseMatrixCSR& A,
                                      const DistributedVector& b,
                                      const DistributedVector& x)
    {
        const Scalar denominator = norm2_mpi(b);
        const Scalar numerator = residual_norm_mpi(A, b, x);

        if (denominator == 0.0)
        {
            return numerator;
        }

        return numerator / denominator;
    }

    Scalar forward_error_norm_mpi(const DistributedVector& x,
                                  const DistributedVector& x_true)
    {
        DistributedVector error = x;

        axpy_local(-1.0, x_true, error);

        return norm2_mpi(error);
    }

    Scalar relative_forward_error_mpi(const DistributedVector& x,
                                      const DistributedVector& x_true)
    {
        const Scalar denominator = norm2_mpi(x_true);
        const Scalar numerator = forward_error_norm_mpi(x, x_true);

        if (denominator == 0.0)
        {
            return numerator;
        }

        return numerator / denominator;
    }

    bool has_converged_absolute_residual_mpi(Scalar residual,
                                             Scalar tolerance)
    {
        return residual <= tolerance;
    }

    bool has_converged_relative_residual_mpi(Scalar relative_residual,
                                             Scalar tolerance)
    {
        return relative_residual <= tolerance;
    }
}
