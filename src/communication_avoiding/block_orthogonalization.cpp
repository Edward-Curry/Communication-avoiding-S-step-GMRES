#include "communication_avoiding/block_orthogonalization.hpp"

#include <cblas.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace gmres {

namespace {

void truncate_r_factors(DenseMatrix& R_old, DenseMatrix& R_block, Index accepted)
{
    for (Vector& row : R_old) {
        row.resize(accepted);
    }

    R_block.resize(accepted);

    for (Vector& row : R_block) {
        row.resize(accepted);
    }
}

}

BlockOrthogonalizationResult block_modified_gram_schmidt(const DenseBlock& old_basis,
                                                         Index old_cols,
                                                         const DenseBlock& input_block)
{
    if (input_block.cols() == 0) {
        throw std::invalid_argument("block_modified_gram_schmidt: input_block is empty.");
    }

    if (old_basis.rows() != input_block.rows()) {
        throw std::invalid_argument("block_modified_gram_schmidt: basis and input row counts differ.");
    }

    if (old_cols > old_basis.cols()) {
        throw std::invalid_argument("block_modified_gram_schmidt: old_cols exceeds the basis width.");
    }

    const Index rows = input_block.rows();
    const Index block_size = input_block.cols();
    const int n = static_cast<int>(rows);

    BlockOrthogonalizationResult result;
    result.R_old.assign(old_cols, Vector(block_size, 0.0));
    result.R_block.assign(block_size, Vector(block_size, 0.0));

    DenseBlock Q(rows, block_size);
    Index accepted = 0;

    Vector w(rows);

    for (Index j = 0; j < block_size; ++j) {
        std::copy(input_block.column(j), input_block.column(j) + rows, w.begin());

        // First: orthogonalise this column against the old basis.
        for (Index i = 0; i < old_cols; ++i) {
            const Scalar coefficient = cblas_ddot(n, old_basis.column(i), 1, w.data(), 1);
            result.R_old[i][j] = coefficient;
            cblas_daxpy(n, -coefficient, old_basis.column(i), 1, w.data(), 1);
        }

        // Second: orthogonalise against previous columns in this new block.
        for (Index i = 0; i < accepted; ++i) {
            const Scalar coefficient = cblas_ddot(n, Q.column(i), 1, w.data(), 1);
            result.R_block[i][j] = coefficient;
            cblas_daxpy(n, -coefficient, Q.column(i), 1, w.data(), 1);
        }

        // Third: normalise this new column.
        const Scalar remaining_norm = cblas_dnrm2(n, w.data(), 1);

        if (remaining_norm == 0.0) {
            truncate_r_factors(result.R_old, result.R_block, accepted);
            result.Q_block = Q.leading_columns(accepted);
            result.accepted_columns = accepted;
            result.truncated = true;
            return result;
        }

        result.R_block[j][j] = remaining_norm;
        cblas_dscal(n, 1.0 / remaining_norm, w.data(), 1);
        Q.set_column(j, w);
        ++accepted;
    }

    result.Q_block = std::move(Q);
    result.accepted_columns = accepted;
    result.truncated = false;

    return result;
}

}
