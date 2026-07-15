#ifndef COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_HPP
#define COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_HPP

#include "common/dense_block.hpp"
#include "common/types.hpp"

namespace gmres {

struct BlockOrthogonalizationResult {
    DenseBlock Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

// Orthogonalises input_block against the leading old_cols columns of
// old_basis with column-wise modified Gram-Schmidt. On truncation the
// returned R factors have exactly accepted_columns columns.
BlockOrthogonalizationResult block_modified_gram_schmidt(const DenseBlock& old_basis,
                                                         Index old_cols,
                                                         const DenseBlock& input_block);

}

#endif
