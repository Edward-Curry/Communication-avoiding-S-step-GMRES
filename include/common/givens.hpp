#ifndef COMMON_GIVENS_HPP
#define COMMON_GIVENS_HPP

#include "common/types.hpp"

namespace gmres
{
    void generate_givens(Scalar a,
                         Scalar b,
                         Scalar& c,
                         Scalar& s);

    void apply_givens(Scalar c,
                      Scalar s,
                      Scalar& a,
                      Scalar& b);
}

#endif