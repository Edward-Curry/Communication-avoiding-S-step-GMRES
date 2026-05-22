#include "common/orthogonalization.hpp"
#include "common/vector_ops.hpp"

#include <cassert>
#include <cmath>
#include <print>
#include <vector>

int main()
{
    std::vector<gmres::Vector> basis = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0}
    };

    gmres::Vector w = {1.0, 2.0, 3.0};
    gmres::Vector h;

    gmres::modified_gram_schmidt(basis, w, h);

    assert(h.size() == 3);

    assert(std::abs(h[0] - 1.0) < 1e-12);
    assert(std::abs(h[1] - 2.0) < 1e-12);
    assert(std::abs(h[2] - 3.0) < 1e-12);

    assert(std::abs(w[0] - 0.0) < 1e-12);
    assert(std::abs(w[1] - 0.0) < 1e-12);
    assert(std::abs(w[2] - 1.0) < 1e-12);

    assert(std::abs(gmres::dot(basis[0], w)) < 1e-12);
    assert(std::abs(gmres::dot(basis[1], w)) < 1e-12);
    assert(std::abs(gmres::norm2(w) - 1.0) < 1e-12);

    std::println("Orthogonalization test passed.");

    return 0;
}