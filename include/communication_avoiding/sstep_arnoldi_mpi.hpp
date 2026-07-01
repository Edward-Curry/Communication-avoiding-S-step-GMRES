#ifndef COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_MPI_HPP
#define COMMUNICATION_AVOIDING_SSTEP_ARNOLDI_MPI_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "communication_avoiding/block_orthogonalization_mpi.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

struct SStepArnoldiMPIResult {
    DistributedVectorList Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

SStepArnoldiMPIResult sstep_arnoldi_block_mpi(const DistributedSparseMatrixCSR& A,
                                              const DistributedVectorList& old_basis,
                                              Index s,
                                              PolynomialBasisType basis_type,
                                              BlockOrthogonalizationMethod method =
                                                  BlockOrthogonalizationMethod::ModifiedGramSchmidt);

}

#endif
