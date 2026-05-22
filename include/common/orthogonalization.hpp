#ifndef COMMON_ORTHOGONALIZATION_HPP
#define COMMON_ORTHOGONALIZATION_HPP

#include "common/types.hpp"

#include <vector>

namespace gmres
{
    void modified_gram_schmidt(const std::vector<Vector>& basis,
                               Vector& w,
                               Vector& h);
}

#endif