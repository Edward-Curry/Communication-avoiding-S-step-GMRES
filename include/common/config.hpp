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
        bool verbose = true;
    };
}

#endif