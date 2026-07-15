#include "communication_avoiding/sstep_arnoldi_mpi.hpp"

#include "communication_avoiding/bcgs2_cholqr_mpi.hpp"
#include "communication_avoiding/krylov_basis_mpi.hpp"

#include <stdexcept>
#include <utility>

namespace gmres {

SStepArnoldiMPIResult sstep_arnoldi_block_mpi(const DistributedSparseMatrixCSR& A,
                                              const DistributedDenseBlock& basis,
                                              Index basis_cols,
                                              Index s,
                                              PolynomialBasisType basis_type,
                                              BlockOrthogonalizationMethod method)
{
    if (basis_cols == 0) {
        throw std::invalid_argument("sstep_arnoldi_block_mpi: old_basis is empty.");
    }

    if (basis_cols > basis.cols()) {
        throw std::invalid_argument("sstep_arnoldi_block_mpi: basis_cols exceeds the basis width.");
    }

    if (s == 0) {
        throw std::invalid_argument("sstep_arnoldi_block_mpi: s must be positive.");
    }

    const DistributedVector q(basis.global_rows(),
                              basis.local_start(),
                              basis.local_block().get_column(basis_cols - 1),
                              basis.communicator());

    DistributedDenseBlock krylov_block =
        generate_krylov_block_mpi(A, q, s, basis_type);

    BlockOrthogonalizationMPIResult ortho_result;

    switch (method) {
    case BlockOrthogonalizationMethod::ModifiedGramSchmidt:
        ortho_result = block_modified_gram_schmidt_mpi(basis, basis_cols, krylov_block);
        break;
    case BlockOrthogonalizationMethod::BCGS2CholQR:
        ortho_result = bcgs2_cholqr_mpi(basis, basis_cols, krylov_block);
        break;
    }

    SStepArnoldiMPIResult result;
    result.Q_block = std::move(ortho_result.Q_block);
    result.R_old = std::move(ortho_result.R_old);
    result.R_block = std::move(ortho_result.R_block);
    result.accepted_columns = ortho_result.accepted_columns;
    result.truncated = ortho_result.truncated;

    return result;
}

}
