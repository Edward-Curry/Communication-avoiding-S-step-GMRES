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

struct SStepArnoldiMPIResult {
    // Local rows of the orthonormal block; the distribution matches the basis.
    DenseBlock Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

// Generates and orthogonalises one s-step Krylov block, starting from the
// last of the leading basis_cols columns of basis.
SStepArnoldiMPIResult sstep_arnoldi_block_mpi(const DistributedSparseMatrixCSR& A,
                                              const DistributedDenseBlock& basis,
                                              Index basis_cols,
                                              Index s,
                                              PolynomialBasisType basis_type,
                                              BlockOrthogonalizationMethod method =
                                                  BlockOrthogonalizationMethod::ModifiedGramSchmidt,
                                              const PartialCholeskyOptions& partial_cholesky_options =
                                                  PartialCholeskyOptions{});

}

#endif
