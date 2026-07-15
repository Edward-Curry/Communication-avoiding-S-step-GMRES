#include "communication_avoiding/sstep_arnoldi.hpp"

#include "communication_avoiding/bcgs2_cholqr.hpp"
#include "communication_avoiding/krylov_basis.hpp"

#include <stdexcept>
#include <utility>

namespace gmres {

SStepArnoldiResult sstep_arnoldi_block(const SparseMatrixCSR& A,
                                       const DenseBlock& basis,
                                       Index basis_cols,
                                       Index s,
                                       PolynomialBasisType basis_type,
                                       BlockOrthogonalizationMethod method,
                                       const PartialCholeskyOptions& partial_cholesky_options)
{
    if (basis_cols == 0) {
        throw std::invalid_argument("sstep_arnoldi_block: old_basis is empty.");
    }

    if (basis_cols > basis.cols()) {
        throw std::invalid_argument("sstep_arnoldi_block: basis_cols exceeds the basis width.");
    }

    if (s == 0) {
        throw std::invalid_argument("sstep_arnoldi_block: s must be positive.");
    }

    const Vector q = basis.get_column(basis_cols - 1);

    DenseBlock krylov_block = generate_krylov_block(A, q, s, basis_type);

    BlockOrthogonalizationResult ortho_result;

    switch (method) {
    case BlockOrthogonalizationMethod::ModifiedGramSchmidt:
        ortho_result = block_modified_gram_schmidt(basis, basis_cols, krylov_block);
        break;
    case BlockOrthogonalizationMethod::BCGS2CholQR:
        ortho_result = bcgs2_cholqr(basis,
                                    basis_cols,
                                    krylov_block,
                                    partial_cholesky_options);
        break;
    }

    SStepArnoldiResult result;
    result.Q_block = std::move(ortho_result.Q_block);
    result.R_old = std::move(ortho_result.R_old);
    result.R_block = std::move(ortho_result.R_block);
    result.accepted_columns = ortho_result.accepted_columns;
    result.truncated = ortho_result.truncated;

    return result;
}

}
