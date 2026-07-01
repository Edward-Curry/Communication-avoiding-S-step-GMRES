#include "communication_avoiding/partial_cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace gmres {

PartialCholeskyResult partial_cholesky(const DenseBlock& gram)
{
    if (gram.rows() != gram.cols()) {
        throw std::invalid_argument("Partial Cholesky requires a square Gram matrix.");
    }

    const Index size = gram.rows();
    DenseBlock factor(size, size);

    Scalar largest_diagonal = 0.0;
    for (Index i = 0; i < size; ++i) {
        largest_diagonal = std::max(largest_diagonal, std::abs(gram(i, i)));
    }

    const Scalar pivot_tolerance =
        std::numeric_limits<Scalar>::epsilon()
        * static_cast<Scalar>(std::max<Index>(size, 1))
        * largest_diagonal;

    Index accepted = 0;

    for (Index col = 0; col < size; ++col) {
        for (Index row = 0; row < col; ++row) {
            Scalar value = gram(row, col);

            for (Index k = 0; k < row; ++k) {
                value -= factor(k, row) * factor(k, col);
            }

            if (factor(row, row) == 0.0) {
                break;
            }

            factor(row, col) = value / factor(row, row);
        }

        Scalar pivot = gram(col, col);
        for (Index k = 0; k < col; ++k) {
            pivot -= factor(k, col) * factor(k, col);
        }

        if (!std::isfinite(pivot) || pivot <= pivot_tolerance) {
            break;
        }

        factor(col, col) = std::sqrt(pivot);
        accepted = col + 1;
    }

    PartialCholeskyResult result;
    result.R = factor.leading_principal_block(accepted);
    result.accepted_columns = accepted;
    result.truncated = accepted < size;
    return result;
}

}
