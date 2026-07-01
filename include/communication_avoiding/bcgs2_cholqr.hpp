#ifndef COMMUNICATION_AVOIDING_BCGS2_CHOLQR_HPP
#define COMMUNICATION_AVOIDING_BCGS2_CHOLQR_HPP

#include "communication_avoiding/block_orthogonalization.hpp"

namespace gmres {

BlockOrthogonalizationResult bcgs2_cholqr(
    const VectorList& old_basis,
    const VectorList& input_block);

}

#endif
