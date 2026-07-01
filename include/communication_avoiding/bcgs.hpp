#ifndef COMMUNICATION_AVOIDING_BCGS_HPP
#define COMMUNICATION_AVOIDING_BCGS_HPP

#include "common/dense_block.hpp"

namespace gmres {

struct BCGSPassResult {
    DenseBlock block;
    DenseBlock coefficients;
};

BCGSPassResult bcgs_pass(const DenseBlock& old_basis,
                         const DenseBlock& input_block);

}

#endif
