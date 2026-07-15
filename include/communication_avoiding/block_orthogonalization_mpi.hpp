#ifndef COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_MPI_HPP
#define COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_MPI_HPP

#include "common/dense_block.hpp"
#include "common/types.hpp"
#include "parallel/distributed_dense_block.hpp"

namespace gmres {

struct BlockOrthogonalizationMPIResult {
    // Local rows of the orthonormal block; the distribution matches the input.
    DenseBlock Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

// Orthogonalises input_block against the leading old_cols columns of
// old_basis with column-wise modified Gram-Schmidt (one MPI_Allreduce per
// coefficient; this is the non-communication-avoiding reference path).
// On truncation the returned R factors have exactly accepted_columns columns.
BlockOrthogonalizationMPIResult block_modified_gram_schmidt_mpi(
    const DistributedDenseBlock& old_basis,
    Index old_cols,
    const DistributedDenseBlock& input_block
);

}

#endif
