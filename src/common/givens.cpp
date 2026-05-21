#include "common/givens.hpp"

#include <cmath>

namespace gmres
{
    void generate_givens(Scalar a,
                         Scalar b,
                         Scalar& c,
                         Scalar& s)
    {
        if (b == 0.0)
        {
            c = 1.0;
            s = 0.0;
        }
        else
        {
            Scalar r = std::sqrt(a * a + b * b);

            c = a / r;
            s = b / r;
        }
    }

    void apply_givens(Scalar c,
                      Scalar s,
                      Scalar& a,
                      Scalar& b)
    {
        Scalar temp = c * a + s * b;

        b = -s * a + c * b;
        a = temp;
    }
}