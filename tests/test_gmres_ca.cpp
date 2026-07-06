#include "communication_avoiding/gmres_ca.hpp"

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "common/vector_ops.hpp"
#include "communication_avoiding/hessenberg_assembly.hpp"
#include "communication_avoiding/sstep_arnoldi.hpp"

#include <cassert>
#include <cmath>
#include <print>
#include <vector>

namespace {

bool nearly_equal(gmres::Scalar a, gmres::Scalar b, gmres::Scalar tolerance)
{
    return std::abs(a - b) < tolerance;
}

void test_hessenberg_assembly(
    gmres::BlockOrthogonalizationMethod method)
{
    using namespace gmres;

    const SparseMatrixCSR A(
        5,
        5,
        Vector{
            2.0, 1.0,
            1.0, 3.0, 1.0,
            1.0, 4.0, 1.0,
            1.0, 5.0, 1.0,
            1.0, 6.0
        },
        std::vector<Index>{
            0, 1,
            0, 1, 2,
            1, 2, 3,
            2, 3, 4,
            3, 4
        },
        std::vector<Index>{0, 2, 5, 8, 11, 13}
    );

    VectorList basis{Vector{1.0, 0.0, 0.0, 0.0, 0.0}};
    DenseMatrix hessenberg(1);

    for (Index block = 0; block < 2; ++block) {
        const SStepArnoldiResult result =
            sstep_arnoldi_block(A,
                                basis,
                                2,
                                PolynomialBasisType::Monomial,
                                method);

        assert(!result.truncated);
        assert(result.accepted_columns == 2);

        append_monomial_hessenberg_block(hessenberg,
                                          result.R_old,
                                          result.R_block);

        for (const Vector& q : result.Q_block) {
            basis.push_back(q);
        }
    }

    assert(hessenberg.size() == 5);
    assert(hessenberg.front().size() == 4);

    for (Index col = 0; col < 4; ++col) {
        const Vector Aq = A.multiply(basis[col]);

        for (Index row = 0; row < 5; ++row) {
            const Scalar explicit_entry = dot(basis[row], Aq);
            assert(nearly_equal(hessenberg[row][col],
                                explicit_entry,
                                1e-9));
        }
    }
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
        test_hessenberg_assembly(method);

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
