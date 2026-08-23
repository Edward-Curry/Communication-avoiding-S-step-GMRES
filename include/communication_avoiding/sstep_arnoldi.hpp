/**
 * @file include/communication_avoiding/sstep_arnoldi.hpp
 * @brief Declares sequential s-step Arnoldi blocks.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_HPP
#define COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_HPP

#include "common/config.hpp"
#include "common/dense_block.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/block_orthogonalization.hpp"
#include "communication_avoiding/partial_cholesky.hpp"
#include "communication_avoiding/polynomial_basis.hpp"

namespace gmres {

/**
 * @brief Stores one sequential s-step Arnoldi block.
 */
struct SStepArnoldiResult {
    DenseBlock Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
    /// @brief Newton shifts used by accepted columns.
    Vector used_shifts;
    /// @brief Scaled-Newton factors used by accepted columns.
    Vector used_scales;
};

/**
 * @brief Generates and orthogonalises one sequential s-step block.
 * @param A System matrix.
 * @param basis Current basis storage.
 * @param basis_cols Number of active leading basis columns.
 * @param s Requested block width.
 * @param basis_type Polynomial basis type.
 * @param method Block orthogonalisation method.
 * @param partial_cholesky_options Acceptance rule for CholQR.
 * @param shifts Newton shifts when required.
 * @return Accepted orthonormal block and recurrence data.
 */
SStepArnoldiResult sstep_arnoldi_block(const SparseMatrixCSR& A,
                                       const DenseBlock& basis,
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
