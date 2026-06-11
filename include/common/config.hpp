#ifndef COMMON_CONFIG_HPP
#define COMMON_CONFIG_HPP

#include "common/types.hpp"

namespace gmres
{
    struct GMRESConfig
    {
        Index restart = 30;
        Index max_iterations = 1000;
        Scalar tolerance = 1e-8;
        Index restart_blocks = 6;
        Index s_step = 5;
        bool verbose = true;
    };
}

#endif