/**
 * @file include/parallel/distributed_utils.hpp
 * @brief Declares distributed residual and error utilities.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_DISTRIBUTED_UTILS_HPP
#define PARALLEL_DISTRIBUTED_UTILS_HPP

#include "common/io.hpp"
#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    /**
     * @brief Computes a distributed algebraic residual.
     * @param A Distributed system matrix.
     * @param b Distributed right-hand side.
     * @param x Distributed approximate solution.
     * @return Distributed vector b minus A times x.
     */
    DistributedVector compute_residual_mpi(const DistributedSparseMatrixCSR& A,
                                           const DistributedVector& b,
                                           const DistributedVector& x);

    /**
     * @brief Computes a global Euclidean residual norm.
     * @param A Distributed system matrix.
     * @param b Distributed right-hand side.
     * @param x Distributed approximate solution.
     * @return Global norm of b minus A times x.
     */
    Scalar residual_norm_mpi(const DistributedSparseMatrixCSR& A,
                             const DistributedVector& b,
                             const DistributedVector& x);

    /**
     * @brief Computes a global relative residual norm.
     * @param A Distributed system matrix.
     * @param b Distributed right-hand side.
     * @param x Distributed approximate solution.
     * @return Global residual norm divided by norm of b.
     */
    Scalar relative_residual_norm_mpi(const DistributedSparseMatrixCSR& A,
                                      const DistributedVector& b,
                                      const DistributedVector& x);

    /**
     * @brief Computes a global forward error norm.
     * @param x Distributed approximate solution.
     * @param x_true Distributed reference solution.
     * @return Global norm of x minus x_true.
     */
    Scalar forward_error_norm_mpi(const DistributedVector& x,
                                  const DistributedVector& x_true);

    /**
     * @brief Computes a global relative forward error.
     * @param x Distributed approximate solution.
     * @param x_true Distributed reference solution.
     * @return Global forward error divided by norm of x_true.
     */
    Scalar relative_forward_error_mpi(const DistributedVector& x,
                                      const DistributedVector& x_true);

    /**
     * @brief Tests an absolute residual tolerance.
     * @param residual Global residual norm.
     * @param tolerance Absolute tolerance.
     * @return True when residual is at most tolerance.
     */
    bool has_converged_absolute_residual_mpi(Scalar residual,
                                             Scalar tolerance);

    /**
     * @brief Tests a relative residual tolerance.
     * @param relative_residual Global relative residual norm.
     * @param tolerance Relative tolerance.
     * @return True when relative_residual is at most tolerance.
     */
    bool has_converged_relative_residual_mpi(Scalar relative_residual,
                                             Scalar tolerance);

    /**
     * @brief Aggregates cached-halo sizes over all ranks.
     * @param A Distributed matrix with an initialized halo plan.
     * @return Summary populated on rank zero.
     */
    HaloExchangeSummary halo_exchange_summary_mpi(
        const DistributedSparseMatrixCSR& A);
}

#endif
