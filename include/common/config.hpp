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

    struct GMRESConfig
    {
        Index restart = 30;
        Index max_iterations = 100000;
        Scalar tolerance = 1e-10;
        Index restart_blocks = 6;
        Index s_step = 5;
        BlockOrthogonalizationMethod block_orthogonalization =
            BlockOrthogonalizationMethod::BCGS2CholQR;
        bool verbose = false;
    };
}

#endif
