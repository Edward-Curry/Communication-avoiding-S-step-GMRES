/**
 * @file include/common/utils.hpp
 * @brief Declares residual and error utilities.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

namespace gmres
{
    /**
     * @brief Computes the algebraic residual.
     * @param A System matrix.
     * @param b Right-hand side.
     * @param x Approximate solution.
     * @return b minus A times x.
     */
    Vector compute_residual(const SparseMatrixCSR& A,
                            const Vector& b,
                            const Vector& x);

    /**
     * @brief Computes the Euclidean residual norm.
     * @param A System matrix.
     * @param b Right-hand side.
     * @param x Approximate solution.
     * @return Norm of b minus A times x.
     */
    Scalar residual_norm(const SparseMatrixCSR& A,
                         const Vector& b,
                         const Vector& x);

    /**
     * @brief Computes the residual norm relative to the right-hand side.
     * @param A System matrix.
     * @param b Right-hand side.
     * @param x Approximate solution.
     * @return Residual norm divided by the norm of b.
     */
    Scalar relative_residual_norm(const SparseMatrixCSR& A,
                                  const Vector& b,
                                  const Vector& x);

    /**
     * @brief Computes the Euclidean forward error.
     * @param x Approximate solution.
     * @param x_true Reference solution.
     * @return Norm of x minus x_true.
     */
    Scalar forward_error_norm(const Vector& x,
                              const Vector& x_true);

    /**
     * @brief Computes the relative forward error.
     * @param x Approximate solution.
     * @param x_true Reference solution.
     * @return Forward error divided by the norm of x_true.
     */
    Scalar relative_forward_error(const Vector& x,
                                  const Vector& x_true);

    /**
     * @brief Tests an absolute residual tolerance.
     * @param residual Residual norm.
     * @param tolerance Absolute tolerance.
     * @return True when residual is at most tolerance.
     */
    bool has_converged_absolute_residual(Scalar residual,
                                         Scalar tolerance);

    /**
     * @brief Tests a relative residual tolerance.
     * @param relative_residual Relative residual norm.
     * @param tolerance Relative tolerance.
     * @return True when relative_residual is at most tolerance.
     */
    bool has_converged_relative_residual(Scalar relative_residual,
                                         Scalar tolerance);
}

#endif
