#ifndef COMMUNICATION_AVOIDING_PARTIAL_CHOLESKY_HPP
#define COMMUNICATION_AVOIDING_PARTIAL_CHOLESKY_HPP

#include "common/config.hpp"
#include "common/dense_block.hpp"

namespace gmres {

struct PartialCholeskyOptions {
    PartialCholeskyStoppingRule stopping_rule =
        PartialCholeskyStoppingRule::TriangularConditionEstimate;
    Scalar condition_limit = 1e7;
};

struct PartialCholeskyResult {
    DenseBlock R;
    Index accepted_columns = 0;
    bool truncated = false;
};

PartialCholeskyResult partial_cholesky(
    const DenseBlock& gram,
    const PartialCholeskyOptions& options = {});

}

#endif
