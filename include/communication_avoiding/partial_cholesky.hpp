/**
 * @file include/communication_avoiding/partial_cholesky.hpp
 * @brief Declares condition-limited partial Cholesky factorisation.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_PARTIAL_CHOLESKY_HPP
#define COMMUNICATION_AVOIDING_PARTIAL_CHOLESKY_HPP

#include "common/config.hpp"
#include "common/dense_block.hpp"

namespace gmres {

/**
 * @brief Configures partial-Cholesky acceptance.
 */
struct PartialCholeskyOptions {
    /// @brief Rule used to stop the leading factorisation.
    PartialCholeskyStoppingRule stopping_rule =
        PartialCholeskyStoppingRule::TriangularConditionEstimate;
    /// @brief Maximum accepted triangular-factor condition estimate.
    Scalar condition_limit = 1e7;
};

/**
 * @brief Stores an accepted leading Cholesky factor.
 */
struct PartialCholeskyResult {
    DenseBlock R;
    Index accepted_columns = 0;
    bool truncated = false;
};

/**
 * @brief Factors the stable leading part of a Gram matrix.
 * @param gram Symmetric Gram matrix.
 * @param options Partial-Cholesky acceptance options.
 * @return Triangular factor, accepted width, and truncation status.
 */
PartialCholeskyResult partial_cholesky(
    const DenseBlock& gram,
    const PartialCholeskyOptions& options = {});

}

#endif
