/**
 * @file include/parallel/distributed_vector_ops.hpp
 * @brief Declares distributed dense vector operations.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_DISTRIBUTED_VECTOR_OPS_HPP
#define PARALLEL_DISTRIBUTED_VECTOR_OPS_HPP

#include "common/types.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    /**
     * @brief Validates matching distributed-vector layouts.
     * @param x First vector.
     * @param y Second vector.
     */
    void check_compatible(const DistributedVector& x,
                          const DistributedVector& y);

    /**
     * @brief Computes a global Euclidean inner product.
     * @param x First distributed vector.
     * @param y Second distributed vector.
     * @return Global dot product of x and y.
     */
    Scalar dot_mpi(const DistributedVector& x,
                   const DistributedVector& y);

    /**
     * @brief Computes a global Euclidean norm.
     * @param x Distributed vector.
     * @return Global two-norm of x.
     */
    Scalar norm2_mpi(const DistributedVector& x);

    /**
     * @brief Scales local vector values in place.
     * @param alpha Scale factor.
     * @param x Distributed vector updated locally.
     */
    void scal_local(Scalar alpha,
                    DistributedVector& x);

    /**
     * @brief Adds a scaled distributed vector to another locally.
     * @param alpha Scale factor.
     * @param x Distributed source vector.
     * @param y Distributed target updated locally.
     */
    void axpy_local(Scalar alpha,
                    const DistributedVector& x,
                    DistributedVector& y);
}

#endif
