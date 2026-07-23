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

struct SStepArnoldiResult {
    DenseBlock Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
    // Shift/scale values actually used for the accepted columns (empty for
    // Monomial). Needed by append_shifted_hessenberg_block for Newton/
    // ScaledNewton; sized to accepted_columns, not the requested width.
    Vector used_shifts;
    Vector used_scales;
};

// Generates and orthogonalises one s-step Krylov block, starting from the
// last of the leading basis_cols columns of basis. shifts is ignored for
// Monomial; required (nonempty) for Newton/ScaledNewton.
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
