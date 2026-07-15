#ifndef COMMON_CONFIG_HPP
#define COMMON_CONFIG_HPP

#include "common/types.hpp"

namespace gmres
{
    enum class BlockOrthogonalizationMethod
    {
        ModifiedGramSchmidt,
        BCGS2CholQR
    };

    enum class PartialCholeskyStoppingRule
    {
        PivotOnly,
        TriangularConditionEstimate
    };

    struct GMRESConfig
    {
        Index restart = 30;
        Index max_iterations = 100000;
        Scalar tolerance = 1e-10;
        Index restart_blocks = 6;
        Index s_step = 5;
        BlockOrthogonalizationMethod block_orthogonalization =
            BlockOrthogonalizationMethod::BCGS2CholQR;
        PartialCholeskyStoppingRule partial_cholesky_stopping_rule =
            PartialCholeskyStoppingRule::TriangularConditionEstimate;
        Scalar partial_cholesky_condition_limit = 1e7;
        bool verbose = false;
    };
}

#endif
