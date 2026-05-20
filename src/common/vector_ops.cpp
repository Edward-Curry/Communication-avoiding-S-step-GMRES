#include "common/vector_ops.hpp"

#include <cmath>
#include <stdexcept>

namespace gmres
{
    void check_same_size(const Vector& x, const Vector& y)
    {
        if (x.size() != y.size())
        {
            throw std::invalid_argument("Vector sizes do not match.");
        }
    }

    Scalar dot(const Vector& x, const Vector& y)
    {
        check_same_size(x, y);

        Scalar result = 0.0;

        for (Index i = 0; i < x.size(); ++i)
        {
            result += x[i] * y[i];
        }

        return result;
    }

    Scalar norm2(const Vector& x)
    {
        return std::sqrt(dot(x, x));
    }

    void scal(Scalar alpha, Vector& x)
    {
        for (Index i = 0; i < x.size(); ++i)
        {
            x[i] *= alpha;
        }
    }

    void axpy(Scalar alpha, const Vector& x, Vector& y)
    {
        check_same_size(x, y);

        for (Index i = 0; i < x.size(); ++i)
        {
            y[i] += alpha * x[i];
        }
    }
}