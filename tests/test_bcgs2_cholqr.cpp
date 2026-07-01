#include "communication_avoiding/bcgs2_cholqr.hpp"

#include "common/vector_ops.hpp"
#include "communication_avoiding/partial_cholesky.hpp"

#include <cassert>
#include <cmath>
#include <print>

namespace {

bool nearly_equal(gmres::Scalar left,
                  gmres::Scalar right,
                  gmres::Scalar tolerance = 1e-11)
{
    return std::abs(left - right) <= tolerance;
}

}

int main()
{
    using namespace gmres;

    VectorList old_basis{Vector{1.0, 0.0, 0.0, 0.0}};
    VectorList input_block{
        Vector{1.0, 1.0, 0.0, 0.0},
        Vector{2.0, 1.0, 1.0, 0.0}
    };

    BlockOrthogonalizationResult result =
        bcgs2_cholqr(old_basis, input_block);

    assert(!result.truncated);
    assert(result.accepted_columns == input_block.size());
    assert(result.Q_block.size() == input_block.size());

    for (const Vector& column : result.Q_block) {
        assert(nearly_equal(norm2(column), 1.0));
        assert(nearly_equal(dot(old_basis.front(), column), 0.0));
    }

    assert(nearly_equal(dot(result.Q_block[0], result.Q_block[1]), 0.0));

    for (Index col = 0; col < input_block.size(); ++col) {
        Vector reconstructed(input_block[col].size(), 0.0);

        for (Index old_col = 0; old_col < old_basis.size(); ++old_col) {
            axpy(result.R_old[old_col][col],
                 old_basis[old_col],
                 reconstructed);
        }

        for (Index block_col = 0; block_col < result.Q_block.size(); ++block_col) {
            axpy(result.R_block[block_col][col],
                 result.Q_block[block_col],
                 reconstructed);
        }

        for (Index row = 0; row < reconstructed.size(); ++row) {
            assert(nearly_equal(reconstructed[row], input_block[col][row]));
        }
    }

    DenseBlock rank_deficient_gram(2, 2);
    rank_deficient_gram(0, 0) = 1.0;
    rank_deficient_gram(0, 1) = 1.0;
    rank_deficient_gram(1, 0) = 1.0;
    rank_deficient_gram(1, 1) = 1.0;

    PartialCholeskyResult partial = partial_cholesky(rank_deficient_gram);
    assert(partial.truncated);
    assert(partial.accepted_columns == 1);

    std::println("BCGS2-CholQR test passed.");
    return 0;
}
