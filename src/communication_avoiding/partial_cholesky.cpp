/**
 * @file src/communication_avoiding/partial_cholesky.cpp
 * @brief Implements condition-limited partial Cholesky factorization.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "communication_avoiding/partial_cholesky.hpp"

#include <algorithm>
#include <cmath>
#include <lapacke.h>
#include <limits>
#include <stdexcept>
#include <utility>

namespace gmres {

namespace {

/**
 * @brief Converts a project dimension to the LAPACK integer type.
 * @param size Dimension to convert.
 * @return LAPACK-compatible dimension.
 */
lapack_int lapack_size(Index size)
{
    if (size > static_cast<Index>(std::numeric_limits<lapack_int>::max())) {
        throw std::length_error("Partial Cholesky block exceeds the LAPACK size range.");
    }

    return static_cast<lapack_int>(size);
}

/**
 * @brief Estimates the one-norm condition number of an upper-triangular factor.
 * @param upper_triangular Square upper-triangular factor.
 * @return Estimated condition number, or infinity for a singular factor.
 */
Scalar estimate_upper_triangular_condition(const DenseBlock& upper_triangular)
{
    if (upper_triangular.rows() != upper_triangular.cols()) {
        throw std::invalid_argument("Condition estimate requires a square triangular matrix.");
    }

    if (upper_triangular.rows() == 0) {
        return 1.0;
    }

    double reciprocal_condition = 0.0;
    const lapack_int size = lapack_size(upper_triangular.rows());

    const lapack_int info = LAPACKE_dtrcon(LAPACK_COL_MAJOR,
                                           '1',
                                           'U',
                                           'N',
                                           size,
                                           upper_triangular.data(),
                                           size,
                                           &reciprocal_condition);

    if (info != 0) {
        throw std::runtime_error("LAPACKE_dtrcon failed while estimating Cholesky condition.");
    }

    if (reciprocal_condition <= 0.0) {
        return std::numeric_limits<Scalar>::infinity();
    }

    return 1.0 / reciprocal_condition;
}

}

PartialCholeskyResult partial_cholesky(const DenseBlock& gram,
                                       const PartialCholeskyOptions& options)
{
    if (gram.rows() != gram.cols()) {
        throw std::invalid_argument("Partial Cholesky requires a square Gram matrix.");
    }

    const Index size = gram.rows();

    // The usable prefix ends at the first invalid Gram diagonal.
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

    // Factor the symmetrically scaled Gram matrix.
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

        if (options.stopping_rule
            == PartialCholeskyStoppingRule::TriangularConditionEstimate) {
            const DenseBlock candidate =
                factor.leading_principal_block(col + 1);
            const Scalar condition =
                estimate_upper_triangular_condition(candidate);

            if (condition > options.condition_limit) {
                break;
            }
        }

        accepted = col + 1;
    }

    // Undo the column scaling in the accepted factor.
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
