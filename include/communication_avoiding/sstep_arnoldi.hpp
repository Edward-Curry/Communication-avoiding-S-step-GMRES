#ifndef COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_HPP
#define COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_HPP

#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/block_orthogonalization.hpp"
#include "communication_avoiding/polynomial_basis.hpp"

namespace gmres {

struct SStepArnoldiResult {
    VectorList Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

SStepArnoldiResult sstep_arnoldi_block(const SparseMatrixCSR& A,
                                       const VectorList& old_basis,
                                       Index s,
                                       PolynomialBasisType basis_type);

}

#endif