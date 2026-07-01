#include "communication_avoiding/gmres_ca.hpp"

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "common/vector_ops.hpp"

#include <cassert>
#include <cmath>
#include <print>
#include <vector>

namespace {

bool nearly_equal(gmres::Scalar a, gmres::Scalar b, gmres::Scalar tolerance)
{
    return std::abs(a - b) < tolerance;
}

}

int main()
{
    using namespace gmres;

    /*
        Matrix:

        [4 1]
        [1 3]

        Stored in CSR format.
    */
    SparseMatrixCSR A(
        2,
        2,
        Vector{4.0, 1.0, 1.0, 3.0},
        std::vector<Index>{0, 1, 0, 1},
        std::vector<Index>{0, 2, 4}
    );

    Vector b{6.0, 7.0};
    Vector x0{0.0, 0.0};

    GMRESConfig config;
    config.restart_blocks = 2;
    config.s_step = 1;
    config.max_iterations = 100;
    config.tolerance = 1e-8;
    config.verbose = true;

    const std::vector<BlockOrthogonalizationMethod> methods{
        BlockOrthogonalizationMethod::BCGS2CholQR,
        BlockOrthogonalizationMethod::ModifiedGramSchmidt
    };

    for (BlockOrthogonalizationMethod method : methods) {
        config.block_orthogonalization = method;
        CAGMRESResult result = gmres_ca(A, b, x0, config);

        assert(result.converged);
        assert(result.x.size() == 2);
        assert(nearly_equal(result.x[0], 1.0, 1e-8));
        assert(nearly_equal(result.x[1], 2.0, 1e-8));

        Vector residual = b;
        Vector Ax = A.multiply(result.x);
        axpy(-1.0, Ax, residual);
        assert(norm2(residual) < 1e-8);
    }

    std::println("test_gmres_ca passed for both block orthogonalization methods.");

    return 0;
}
