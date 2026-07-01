#ifndef COMMUNICATION_AVOIDING_CHOLQR_HPP
#define COMMUNICATION_AVOIDING_CHOLQR_HPP

#include "common/dense_block.hpp"

namespace gmres {

struct CholQRResult {
    DenseBlock Q;
    DenseBlock R;
    Index accepted_columns = 0;
    bool truncated = false;
};

CholQRResult cholqr(const DenseBlock& input_block);

}

#endif
