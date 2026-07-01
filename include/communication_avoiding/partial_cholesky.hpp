#ifndef COMMUNICATION_AVOIDING_PARTIAL_CHOLESKY_HPP
#define COMMUNICATION_AVOIDING_PARTIAL_CHOLESKY_HPP

#include "common/dense_block.hpp"

namespace gmres {

struct PartialCholeskyResult {
    DenseBlock R;
    Index accepted_columns = 0;
    bool truncated = false;
};

PartialCholeskyResult partial_cholesky(const DenseBlock& gram);

}

#endif
