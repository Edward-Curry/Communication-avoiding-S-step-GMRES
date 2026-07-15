#include "communication_avoiding/partial_cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gmres {

PartialCholeskyResult partial_cholesky(const DenseBlock& gram)
{
    if (gram.rows() != gram.cols()) {
        throw std::invalid_argument("Partial Cholesky requires a square Gram matrix.");
    }

    const Index size = gram.rows();

    // Column scales from the Gram diagonal. Columns after the first
    // non-positive or non-finite diagonal are unusable, so the usable
    // prefix ends there.
    Vector scale(size, 0.0);
    Index limit = size;

    for (Index i = 0; i < size; ++i) {
        const Scalar diagonal = gram(i, i);

        if (!std::isfinite(diagonal) || diagonal <= 0.0) {
            limit = i;
            break;
        }

        scale[i] = std::sqrt(diagonal);
    }

    // Factor the symmetrically scaled Gram matrix D^{-1} G D^{-1}, whose
    // diagonal is 1. Monomial Krylov columns can differ in norm by many
    // orders of magnitude; without this scaling the pivot test rejects
    // columns that are merely badly scaled rather than nearly dependent.
    DenseBlock factor(size, size);

    const Scalar pivot_tolerance =
        std::numeric_limits<Scalar>::epsilon()
        * static_cast<Scalar>(std::max<Index>(size, 1));

    Index accepted = 0;

    for (Index col = 0; col < limit; ++col) {
        for (Index row = 0; row < col; ++row) {
            Scalar value = gram(row, col) / (scale[row] * scale[col]);

            for (Index k = 0; k < row; ++k) {
                value -= factor(k, row) * factor(k, col);
            }

            factor(row, col) = value / factor(row, row);
        }

        Scalar pivot = 1.0;
        for (Index k = 0; k < col; ++k) {
            pivot -= factor(k, col) * factor(k, col);
        }

        if (!std::isfinite(pivot) || pivot <= pivot_tolerance) {
            break;
        }

        factor(col, col) = std::sqrt(pivot);
        accepted = col + 1;
    }

    // Undo the column scaling: G = D (D^{-1} G D^{-1}) D implies R = R_scaled D.
    DenseBlock unscaled(accepted, accepted);

    for (Index col = 0; col < accepted; ++col) {
        for (Index row = 0; row <= col; ++row) {
            unscaled(row, col) = factor(row, col) * scale[col];
        }
    }

    PartialCholeskyResult result;
    result.R = std::move(unscaled);
    result.accepted_columns = accepted;
    result.truncated = accepted < size;
    return result;
}

}
