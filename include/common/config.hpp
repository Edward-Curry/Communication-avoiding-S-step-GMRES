#ifndef COMMON_CONFIG_HPP
#define COMMON_CONFIG_HPP

#include "common/types.hpp"

namespace gmres
{
    enum class BlockOrthogonalizationMethod
    {
        ModifiedGramSchmidt,
        BCGS2CholQR
    };

    enum class PartialCholeskyStoppingRule
    {
        PivotOnly,
        TriangularConditionEstimate
    };

    struct GMRESConfig
    {
        Index restart = 30;
        Index max_iterations = 100000;
        Scalar tolerance = 1e-10;
        Index restart_blocks = 6;
        Index s_step = 5;

        // Adaptive s-step selection. When adaptive_s is false the block width
        // is fixed at s_step (unchanged behavior). When true, the width starts
        // at s_step (clamped into [s_min, s_max]) and adapts per block: it
        // shrinks toward the accepted width after a truncated/ill-conditioned
        // block, and grows by one after s_grow_after consecutive fully
        // accepted blocks. The width is always kept within [s_min, s_max].
        bool adaptive_s = true;
        Index s_min = 1;
        Index s_max = 10;
        Index s_grow_after = 2;

        // Initial-s estimator (requires adaptive_s). The first block of every
        // restart cycle requests s_max columns and the condition-limited
        // partial Cholesky decides how many are stable; the accepted count
        // becomes the working width for the rest of the cycle (further
        // adapted by the rules above). The probe costs no extra communication
        // and its accepted columns are kept as regular basis vectors; only
        // (s_max - accepted) SpMVs are wasted when the matrix cannot support
        // s_max. When enabled, s_step no longer sets the starting width.
        bool s_initial_probe = true;

        BlockOrthogonalizationMethod block_orthogonalization =
            BlockOrthogonalizationMethod::BCGS2CholQR;
        PartialCholeskyStoppingRule partial_cholesky_stopping_rule =
            PartialCholeskyStoppingRule::TriangularConditionEstimate;
        Scalar partial_cholesky_condition_limit = 1e7;
        bool verbose = false;
    };
}

#endif
