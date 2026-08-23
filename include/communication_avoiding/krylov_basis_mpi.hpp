/**
 * @file include/communication_avoiding/krylov_basis_mpi.hpp
 * @brief Declares MPI s-step Krylov-block generation.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_KRYLOV_BASIS_MPI_HPP
#define COMMUNICATION_AVOIDING_KRYLOV_BASIS_MPI_HPP

#include "common/types.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

/**
 * @brief Stores one generated distributed Krylov block.
 */
struct KrylovBlockMPIResult {
    DistributedDenseBlock block;
    /// @brief Replicated Scaled-Newton factors, when used.
    Vector column_scales;
};

/**
 * @brief Generates one distributed s-step Krylov block.
 * @param A Distributed system matrix.
 * @param q Distributed seed vector.
 * @param s Requested block width.
 * @param basis_type Polynomial basis type.
 * @param shifts Replicated Leja-ordered Newton shifts when required.
 * @return Generated distributed block and any Scaled-Newton factors.
 */
KrylovBlockMPIResult generate_krylov_block_mpi(const DistributedSparseMatrixCSR& A,
                                               const DistributedVector& q,
                                               Index s,
                                               PolynomialBasisType basis_type,
                                               const Vector& shifts = Vector());

}

#endif
