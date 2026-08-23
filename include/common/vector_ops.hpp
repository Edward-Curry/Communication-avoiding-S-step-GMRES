/**
 * @file include/common/vector_ops.hpp
 * @brief Declares dense vector operations.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMON_VECTOR_OPS_HPP
#define COMMON_VECTOR_OPS_HPP

#include "common/types.hpp"

namespace gmres
{
    /**
     * @brief Validates equal vector lengths.
     * @param x First vector.
     * @param y Second vector.
     */
    void check_same_size(const Vector& x, const Vector& y);

    /**
     * @brief Computes a Euclidean inner product.
     * @param x First vector.
     * @param y Second vector.
     * @return Dot product of x and y.
     */
    Scalar dot(const Vector& x, const Vector& y);

    /**
     * @brief Computes a Euclidean vector norm.
     * @param x Input vector.
     * @return Two-norm of x.
     */
    Scalar norm2(const Vector& x);

    /**
     * @brief Scales a vector in place.
     * @param alpha Scale factor.
     * @param x Vector updated to alpha times x.
     */
    void scal(Scalar alpha, Vector& x);

    /**
     * @brief Adds a scaled vector to another vector.
     * @param alpha Scale factor.
     * @param x Source vector.
     * @param y Target vector updated to alpha times x plus y.
     */
    void axpy(Scalar alpha, const Vector& x, Vector& y);
}

#endif
