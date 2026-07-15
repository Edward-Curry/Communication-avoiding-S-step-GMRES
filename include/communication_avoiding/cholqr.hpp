#ifndef COMMUNICATION_AVOIDING_CHOLQR_HPP
#define COMMUNICATION_AVOIDING_CHOLQR_HPP

#include "common/dense_block.hpp"
#include "communication_avoiding/partial_cholesky.hpp"

namespace gmres {

struct CholQRResult {
    DenseBlock Q;
    DenseBlock R;
    Index accepted_columns = 0;
    bool truncated = false;
};

CholQRResult cholqr(
    const DenseBlock& input_block,
    const PartialCholeskyOptions& partial_cholesky_options =
        PartialCholeskyOptions{});

}

#endif
