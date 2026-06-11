#include "communication_avoiding/sstep_arnoldi.hpp"

#include "communication_avoiding/krylov_basis.hpp"

#include <stdexcept>

namespace gmres {

SStepArnoldiResult sstep_arnoldi_block(const SparseMatrixCSR& A,
                                       const VectorList& old_basis,
                                       Index s,
                                       PolynomialBasisType basis_type)
{
    if (old_basis.empty()) {
        throw std::invalid_argument("sstep_arnoldi_block: old_basis is empty.");
    }

    if (s == 0) {
        throw std::invalid_argument("sstep_arnoldi_block: s must be positive.");
    }

    const Vector& q = old_basis.back();

    VectorList krylov_block = generate_krylov_block(A, q, s, basis_type);

    BlockOrthogonalizationResult ortho_result =
        block_modified_gram_schmidt(old_basis, krylov_block);

    SStepArnoldiResult result;
    result.Q_block = ortho_result.Q_block;
    result.R_old = ortho_result.R_old;
    result.R_block = ortho_result.R_block;
    result.accepted_columns = ortho_result.accepted_columns;
    result.truncated = ortho_result.truncated;

    return result;
}

}