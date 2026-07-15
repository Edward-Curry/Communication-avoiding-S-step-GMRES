#ifndef COMMUNICATION_AVOIDING_BCGS_HPP
#define COMMUNICATION_AVOIDING_BCGS_HPP

#include "common/dense_block.hpp"

namespace gmres {

struct BCGSPassResult {
    DenseBlock block;
    DenseBlock coefficients;
};

// Orthogonalises input_block against the leading old_cols columns of
// old_basis, without copying the old basis.
BCGSPassResult bcgs_pass(const DenseBlock& old_basis,
                         Index old_cols,
                         const DenseBlock& input_block);

}

#endif
