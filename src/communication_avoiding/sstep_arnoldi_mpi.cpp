#include "communication_avoiding/sstep_arnoldi_mpi.hpp"

#include "communication_avoiding/krylov_basis_mpi.hpp"

#include <stdexcept>

namespace gmres {

SStepArnoldiMPIResult sstep_arnoldi_block_mpi(const DistributedSparseMatrixCSR& A,
                                              const DistributedVectorList& old_basis,
                                              Index s,
                                              PolynomialBasisType basis_type)
{
    if (old_basis.empty()) {
        throw std::invalid_argument("sstep_arnoldi_block_mpi: old_basis is empty.");
    }

    if (s == 0) {
        throw std::invalid_argument("sstep_arnoldi_block_mpi: s must be positive.");
    }

    const DistributedVector& q = old_basis.back();

    DistributedVectorList krylov_block =
        generate_krylov_block_mpi(A, q, s, basis_type);

    BlockOrthogonalizationMPIResult ortho_result =
        block_modified_gram_schmidt_mpi(old_basis, krylov_block);

    SStepArnoldiMPIResult result;
    result.Q_block = ortho_result.Q_block;
    result.R_old = ortho_result.R_old;
    result.R_block = ortho_result.R_block;
    result.accepted_columns = ortho_result.accepted_columns;
    result.truncated = ortho_result.truncated;

    return result;
}

}