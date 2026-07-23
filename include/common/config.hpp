#ifndef COMMON_CONFIG_HPP
#define COMMON_CONFIG_HPP

#include "common/types.hpp"
#include "communication_avoiding/polynomial_basis.hpp"

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

        // Block recycling across restart cycles. Orthogonal to adaptive_s /
        // s_initial_probe (either, both, or neither may be on at once). After
        // each restart cycle, the recycle_count newly generated blocks with
        // the largest relative residual drop are kept and used to augment the
        // START of the next cycle's search space (GCRO-DR style): the full
        // restart_blocks worth of brand-new blocks is still generated every
        // cycle, unreduced - recycling only adds a head start, never replaces
        // cycle budget. "Keep most recent winner": each cycle's own winners
        // replace the previous cycle's recycled subspace outright (no
        // cross-cycle score comparison, no extra state beyond the columns
        // themselves). Defaults to off until validated.
        bool enable_recycling = false;
        Index recycle_count = 2;

        // Polynomial basis for s-step Krylov block generation. Monomial
        // (Aq,...,A^s q) is the default; Newton/ScaledNewton use shifted
        // products (A-theta_j I)q with theta_j real Ritz-value estimates of
        // A, Leja-ordered for numerical stability, countering the monomial
        // basis's tendency to collapse toward the dominant eigenvector
        // direction as s grows (the diagonal Gram scaling in CholQR already
        // handles the separate column-NORM-imbalance problem, for any basis
        // type). Shifts are computed ONCE, from the first restart cycle's
        // Hessenberg matrix (which necessarily still uses Monomial - there is
        // no spectral information before that), and reused for the rest of
        // the solve. ScaledNewton additionally rescales each generated column
        // by its own norm, guarding against overflow across many shifted
        // products; ordinary Newton does not rescale during generation
        // (CholQR's own diagonal scaling still applies once orthogonalized).
        // Only real Ritz values are used as shifts; a matrix with a genuinely
        // complex spectrum yields fewer usable shifts (falling back toward
        // Monomial for the columns whose real-shift budget runs out).
        PolynomialBasisType polynomial_basis = PolynomialBasisType::Monomial;

        BlockOrthogonalizationMethod block_orthogonalization =
            BlockOrthogonalizationMethod::BCGS2CholQR;
        PartialCholeskyStoppingRule partial_cholesky_stopping_rule =
            PartialCholeskyStoppingRule::TriangularConditionEstimate;
        Scalar partial_cholesky_condition_limit = 1e7;
        bool verbose = false;
    };
}

#endif
