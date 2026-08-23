/**
 * @file include/communication_avoiding/sstep_arnoldi_mpi.hpp
 * @brief Declares MPI s-step Arnoldi blocks.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_MPI_HPP
#define COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_MPI_HPP

#include "common/config.hpp"
#include "common/dense_block.hpp"
#include "common/types.hpp"
#include "communication_avoiding/block_orthogonalization_mpi.hpp"
#include "communication_avoiding/partial_cholesky.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

/**
 * @brief Stores one distributed s-step Arnoldi block.
 */
struct SStepArnoldiMPIResult {
    /// @brief Local rows of the accepted orthonormal block.
    DenseBlock Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
    /// @brief Replicated Newton shifts used by accepted columns.
    Vector used_shifts;
    /// @brief Replicated Scaled-Newton factors used by accepted columns.
    Vector used_scales;
};

/**
 * @brief Generates and orthogonalises one distributed s-step block.
 * @param A Distributed system matrix.
 * @param basis Current distributed basis storage.
 * @param basis_cols Number of active leading basis columns.
 * @param s Requested block width.
 * @param basis_type Polynomial basis type.
 * @param method Block orthogonalisation method.
 * @param partial_cholesky_options Acceptance rule for CholQR.
 * @param shifts Replicated Newton shifts when required.
 * @return Accepted local block and replicated recurrence data.
 */
SStepArnoldiMPIResult sstep_arnoldi_block_mpi(const DistributedSparseMatrixCSR& A,
                                              const DistributedDenseBlock& basis,
                                              Index basis_cols,
                                              Index s,
                                              PolynomialBasisType basis_type,
                                              BlockOrthogonalizationMethod method =
                                                  BlockOrthogonalizationMethod::ModifiedGramSchmidt,
                                              const PartialCholeskyOptions& partial_cholesky_options =
                                                  PartialCholeskyOptions{},
                                              const Vector& shifts = Vector());

}

#endif
