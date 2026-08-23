/**
 * @file include/common/config.hpp
 * @brief Defines solver configuration options.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMON_CONFIG_HPP
#define COMMON_CONFIG_HPP

#include "common/types.hpp"
#include "communication_avoiding/polynomial_basis.hpp"

namespace gmres
{
    /**
     * @brief Selects the block orthogonalisation procedure.
     */
    enum class BlockOrthogonalizationMethod
    {
        ModifiedGramSchmidt,
        BCGS2CholQR
    };

    /**
     * @brief Selects the partial-Cholesky acceptance rule.
     */
    enum class PartialCholeskyStoppingRule
    {
        PivotOnly,
        TriangularConditionEstimate
    };

    /**
     * @brief Stores numerical and algorithmic settings for a GMRES solve.
     */
    struct GMRESConfig
    {
        /// @brief Restart length for conventional GMRES.
        Index restart = 30;
        /// @brief Maximum Krylov iterations permitted by a solve.
        Index max_iterations = 50000;

        /// @brief Relative residual tolerance measured against the initial residual.
        Scalar tolerance = 1e-10;

        /// @brief Maximum s-step blocks generated during one restart cycle.
        Index restart_blocks = 6;
        /// @brief Fixed s-step width when adaptive width control is disabled.
        Index s_step = 5;

        /// @brief Enables block-width adaptation after partial-Cholesky acceptance.
        bool adaptive_s = true;
        /// @brief Smallest block width requested by the adaptive controller.
        Index s_min = 1;
        /// @brief Largest block width requested by the adaptive controller.
        Index s_max = 100;
        /// @brief Full-block count required before increasing the requested width.
        Index s_grow_after = 2;

        /// @brief Probes the initial usable width by requesting the configured maximum.
        bool s_initial_probe = true;

        /// @brief Enables harmonic-Ritz recycling between restart cycles.
        bool enable_recycling = true;
        /// @brief Number of harmonic Ritz vectors retained at a restart.
        Index recycle_count = 2;

        /// @brief Polynomial basis used to generate s-step Krylov blocks.
        PolynomialBasisType polynomial_basis = PolynomialBasisType::Monomial;

        /// @brief Block orthogonalisation method.
        BlockOrthogonalizationMethod block_orthogonalization =
            BlockOrthogonalizationMethod::BCGS2CholQR;
        /// @brief Partial-Cholesky stopping rule.
        PartialCholeskyStoppingRule partial_cholesky_stopping_rule =
            PartialCholeskyStoppingRule::TriangularConditionEstimate;
        /// @brief Maximum accepted triangular-factor condition estimate.
        Scalar partial_cholesky_condition_limit = 1e7;
        /// @brief Enables solver progress output.
        bool verbose = false;
    };
}

#endif
