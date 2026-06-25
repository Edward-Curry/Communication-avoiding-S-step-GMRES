#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

namespace gmres
{
    Vector compute_residual(const SparseMatrixCSR& A,
                            const Vector& b,
                            const Vector& x);

    Scalar residual_norm(const SparseMatrixCSR& A,
                         const Vector& b,
                         const Vector& x);

    Scalar relative_residual_norm(const SparseMatrixCSR& A,
                                  const Vector& b,
                                  const Vector& x);

    Scalar forward_error_norm(const Vector& x,
                              const Vector& x_true);

    Scalar relative_forward_error(const Vector& x,
                                  const Vector& x_true);

    bool has_converged_absolute_residual(Scalar residual,
                                         Scalar tolerance);

    bool has_converged_relative_residual(Scalar relative_residual,
                                         Scalar tolerance);
}

#endif
