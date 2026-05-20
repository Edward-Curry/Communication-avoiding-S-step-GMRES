#include "common/vector_ops.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

int main()
{
    gmres::Vector x = {1.0, 2.0, 3.0};
    gmres::Vector y = {4.0, 5.0, 6.0};

    assert(gmres::dot(x, y) == 32.0);

    double norm = gmres::norm2(x);
    assert(std::abs(norm - std::sqrt(14.0)) < 1e-12);

    gmres::Vector z = x;
    gmres::scal(2.0, z);
    assert(z[0] == 2.0);
    assert(z[1] == 4.0);
    assert(z[2] == 6.0);

    gmres::Vector w = y;
    gmres::axpy(2.0, x, w);
    assert(w[0] == 6.0);
    assert(w[1] == 9.0);
    assert(w[2] == 12.0);

    bool caught_error = false;
    try
    {
        gmres::Vector bad = {1.0, 2.0};
        gmres::dot(x, bad);
    }
    catch (const std::invalid_argument&)
    {
        caught_error = true;
    }

    assert(caught_error);

    std::cout << "Vector operations test passed.\n";

    return 0;
}