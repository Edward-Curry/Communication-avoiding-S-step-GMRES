#ifndef COMMON_VECTOR_OPS_HPP
#define COMMON_VECTOR_OPS_HPP

#include "common/types.hpp"

namespace gmres
{
    void check_same_size(const Vector& x, const Vector& y);

    Scalar dot(const Vector& x, const Vector& y);

    Scalar norm2(const Vector& x);

    void scal(Scalar alpha, Vector& x);

    void axpy(Scalar alpha, const Vector& x, Vector& y);
}

#endif