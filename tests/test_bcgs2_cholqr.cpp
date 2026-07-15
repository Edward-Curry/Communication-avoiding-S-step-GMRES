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

    const DenseBlock old_basis = pack_columns(
        VectorList{Vector{1.0, 0.0, 0.0, 0.0}});
    const DenseBlock input_block = pack_columns(VectorList{
        Vector{1.0, 1.0, 0.0, 0.0},
        Vector{2.0, 1.0, 1.0, 0.0}
    });

    BlockOrthogonalizationResult result =
        bcgs2_cholqr(old_basis, old_basis.cols(), input_block);

    assert(!result.truncated);
    assert(result.accepted_columns == input_block.cols());
    assert(result.Q_block.cols() == input_block.cols());

    for (Index col = 0; col < result.Q_block.cols(); ++col) {
        const Vector column = result.Q_block.get_column(col);
        assert(nearly_equal(norm2(column), 1.0));
        assert(nearly_equal(dot(old_basis.get_column(0), column), 0.0));
    }

    assert(nearly_equal(dot(result.Q_block.get_column(0),
                            result.Q_block.get_column(1)),
                        0.0));

    for (Index col = 0; col < input_block.cols(); ++col) {
        Vector reconstructed(input_block.rows(), 0.0);

        for (Index old_col = 0; old_col < old_basis.cols(); ++old_col) {
            axpy(result.R_old[old_col][col],
                 old_basis.get_column(old_col),
                 reconstructed);
        }

        for (Index block_col = 0; block_col < result.Q_block.cols(); ++block_col) {
            axpy(result.R_block[block_col][col],
                 result.Q_block.get_column(block_col),
                 reconstructed);
        }

        for (Index row = 0; row < reconstructed.size(); ++row) {
            assert(nearly_equal(reconstructed[row], input_block(row, col)));
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

    DenseBlock ill_conditioned_gram(2, 2);
    ill_conditioned_gram(0, 0) = 1.0;
    ill_conditioned_gram(0, 1) = 0.99;
    ill_conditioned_gram(1, 0) = 0.99;
    ill_conditioned_gram(1, 1) = 1.0;

    PartialCholeskyResult pivot_only =
        partial_cholesky(ill_conditioned_gram,
                         {PartialCholeskyStoppingRule::PivotOnly, 2.0});
    assert(!pivot_only.truncated);
    assert(pivot_only.accepted_columns == 2);

    PartialCholeskyResult condition_limited =
        partial_cholesky(
            ill_conditioned_gram,
            {PartialCholeskyStoppingRule::TriangularConditionEstimate, 2.0});
    assert(condition_limited.truncated);
    assert(condition_limited.accepted_columns == 1);

    std::println("BCGS2-CholQR test passed.");
    return 0;
}
