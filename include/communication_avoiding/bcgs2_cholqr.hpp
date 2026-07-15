#ifndef COMMUNICATION_AVOIDING_BCGS2_CHOLQR_HPP
#define COMMUNICATION_AVOIDING_BCGS2_CHOLQR_HPP

#include "common/dense_block.hpp"
#include "communication_avoiding/block_orthogonalization.hpp"

namespace gmres {

// Two-pass block classical Gram-Schmidt with CholQR intra-block
// orthonormalisation, against the leading old_cols columns of old_basis.
BlockOrthogonalizationResult bcgs2_cholqr(
    const DenseBlock& old_basis,
    Index old_cols,
    const DenseBlock& input_block);

}

#endif
