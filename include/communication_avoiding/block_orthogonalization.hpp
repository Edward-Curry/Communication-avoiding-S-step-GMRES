#ifndef COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_HPP
#define COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_HPP

#include "common/types.hpp"

namespace gmres {

struct BlockOrthogonalizationResult {
    VectorList Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

BlockOrthogonalizationResult block_modified_gram_schmidt(const VectorList& old_basis,
                                                         const VectorList& input_block);

}

#endif