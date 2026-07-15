#ifndef COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_HPP
#define COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_HPP

#include "common/config.hpp"
#include "common/dense_block.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/block_orthogonalization.hpp"
#include "communication_avoiding/polynomial_basis.hpp"

namespace gmres {

struct SStepArnoldiResult {
    DenseBlock Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

// Generates and orthogonalises one s-step Krylov block, starting from the
// last of the leading basis_cols columns of basis.
SStepArnoldiResult sstep_arnoldi_block(const SparseMatrixCSR& A,
                                       const DenseBlock& basis,
                                       Index basis_cols,
                                       Index s,
                                       PolynomialBasisType basis_type,
                                       BlockOrthogonalizationMethod method =
                                           BlockOrthogonalizationMethod::ModifiedGramSchmidt);

}

#endif
