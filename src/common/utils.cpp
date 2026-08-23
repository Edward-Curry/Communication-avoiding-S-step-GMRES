/**
 * @file src/common/utils.cpp
 * @brief Implements residual, error, and convergence utilities.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "common/utils.hpp"

#include "common/vector_ops.hpp"

namespace gmres
{
    Vector compute_residual(const SparseMatrixCSR& A,
                            const Vector& b,
                            const Vector& x)
    {
        Vector residual = b;
        Vector Ax = A.multiply(x);

        axpy(-1.0, Ax, residual);

        return residual;
    }

    Scalar residual_norm(const SparseMatrixCSR& A,
                         const Vector& b,
                         const Vector& x)
    {
        return norm2(compute_residual(A, b, x));
    }

    Scalar relative_residual_norm(const SparseMatrixCSR& A,
                                  const Vector& b,
                                  const Vector& x)
    {
        const Scalar denominator = norm2(b);
        const Scalar numerator = residual_norm(A, b, x);

        if (denominator == 0.0)
        {
            return numerator;
        }

        return numerator / denominator;
    }

    Scalar forward_error_norm(const Vector& x,
                              const Vector& x_true)
    {
        Vector error = x;

        axpy(-1.0, x_true, error);

        return norm2(error);
    }

    Scalar relative_forward_error(const Vector& x,
                                  const Vector& x_true)
    {
        const Scalar denominator = norm2(x_true);
        const Scalar numerator = forward_error_norm(x, x_true);

        if (denominator == 0.0)
        {
            return numerator;
        }

        return numerator / denominator;
    }

    bool has_converged_absolute_residual(Scalar residual,
                                         Scalar tolerance)
    {
        return residual <= tolerance;
    }

    bool has_converged_relative_residual(Scalar relative_residual,
                                         Scalar tolerance)
    {
        return relative_residual <= tolerance;
    }
}
