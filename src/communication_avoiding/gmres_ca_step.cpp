#include "communication_avoiding/gmres_ca_step.hpp"

#include "common/dense_block.hpp"
#include "common/givens.hpp"
#include "common/vector_ops.hpp"
#include "communication_avoiding/cholqr.hpp"
#include "communication_avoiding/hessenberg_assembly.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "communication_avoiding/sstep_arnoldi.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gmres {

namespace {

Vector solve_upper_triangular(const DenseMatrix& R, const Vector& g, Index n)
{
    Vector y(n, 0.0);

    for (Index reverse_i = 0; reverse_i < n; ++reverse_i) {
        const Index i = n - 1 - reverse_i;

        Scalar sum = g[i];

        for (Index j = i + 1; j < n; ++j) {
            sum -= R[i][j] * y[j];
        }

        if (R[i][i] == 0.0) {
            throw std::runtime_error("solve_upper_triangular: zero diagonal entry.");
        }

        y[i] = sum / R[i][i];
    }

    return y;
}

// Solves the small k x k upper triangular system T z = w. Used to recover the
// recycled-subspace coefficients: the combined least-squares solve below
// returns w (the coefficient of C = orthonormalize(A U) in the augmented
// system), and since A U = C T, the coefficient of U itself is z = T^{-1} w.
Vector solve_upper_triangular_block(const DenseBlock& T, const Vector& w)
{
    const Index k = T.rows();
    Vector z(k, 0.0);

    for (Index reverse_row = 0; reverse_row < k; ++reverse_row) {
        const Index row = k - 1 - reverse_row;

        Scalar sum = w[row];

        for (Index col = row + 1; col < k; ++col) {
            sum -= T(row, col) * z[col];
        }

        if (T(row, row) == 0.0) {
            throw std::runtime_error("solve_upper_triangular_block: zero diagonal entry.");
        }

        z[row] = sum / T(row, row);
    }

    return z;
}

// Applies the accumulated Givens rotations to Hessenberg columns
// [columns_rotated, total_columns) of H, writing the rotated result into R
// and updating g and the rotation history in place. Used both to fold in
// newly generated s-step blocks and, when a recycled subspace seeds the
// cycle, to fold in that subspace's leading columns before any new block is
// generated - the same incremental Givens-QR process either way.
void fold_hessenberg_columns(const DenseMatrix& H,
                             DenseMatrix& R,
                             Vector& g,
                             Vector& cosines,
                             Vector& sines,
                             Index& columns_rotated,
                             Index total_columns)
{
    for (Index j = columns_rotated; j < total_columns; ++j) {
        for (Index row = 0; row <= j + 1; ++row) {
            R[row][j] = H[row][j];
        }

        for (Index i = 0; i < j; ++i) {
            apply_givens(cosines[i], sines[i], R[i][j], R[i + 1][j]);
        }

        generate_givens(R[j][j], R[j + 1][j], cosines[j], sines[j]);
        apply_givens(cosines[j], sines[j], R[j][j], R[j + 1][j]);
        apply_givens(cosines[j], sines[j], g[j], g[j + 1]);
    }

    columns_rotated = total_columns;
}

struct RecycleCandidate {
    Scalar score = 0.0;
    Index col_start = 0;
    Index col_count = 0;
};

} // namespace

CAGMRESCycleResult gmres_ca_cycle(const SparseMatrixCSR& A,
                                  const Vector& b,
                                  const Vector& x_start,
                                  const Vector& r_start,
                                  Scalar beta,
                                  const GMRESConfig& config,
                                  const DenseBlock& recycled_U,
                                  const Vector& shifts)
{
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("gmres_ca_cycle requires a square matrix.");
    }

    if (b.size() != A.rows()) {
        throw std::invalid_argument("gmres_ca_cycle: b has wrong size.");
    }

    if (x_start.size() != A.cols()) {
        throw std::invalid_argument("gmres_ca_cycle: x_start has wrong size.");
    }

    if (r_start.size() != A.rows()) {
        throw std::invalid_argument("gmres_ca_cycle: r_start has wrong size.");
    }

    if (config.restart_blocks == 0) {
        throw std::invalid_argument("gmres_ca_cycle: restart_blocks must be positive.");
    }

    if (config.s_step == 0) {
        throw std::invalid_argument("gmres_ca_cycle: s_step must be positive.");
    }

    if (config.adaptive_s) {
        if (config.s_min == 0) {
            throw std::invalid_argument("gmres_ca_cycle: s_min must be positive when adaptive_s is set.");
        }

        if (config.s_max < config.s_min) {
            throw std::invalid_argument("gmres_ca_cycle: s_max must be >= s_min when adaptive_s is set.");
        }
    } else if (config.s_initial_probe) {
        throw std::invalid_argument("gmres_ca_cycle: s_initial_probe requires adaptive_s.");
    }

    if (recycled_U.cols() > 0 && recycled_U.rows() != A.rows()) {
        throw std::invalid_argument("gmres_ca_cycle: recycled_U row count does not match the matrix.");
    }

    CAGMRESCycleResult result;
    result.x = x_start;

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    const Index n = A.rows();

    // Worst-case block width per block. Adaptive s can grow up to s_max, so the
    // basis and least-squares buffers must be sized for that upper bound.
    const Index s_cap = config.adaptive_s
        ? std::max(config.s_step, config.s_max)
        : config.s_step;

    // Worst-case recycled-subspace width: each recycled block's own column
    // count is bounded the same way a normal block's is, so recycle_count
    // blocks' worth is a valid upper bound regardless of how recycled_U was
    // produced. Zero when recycling is off, so disabled runs allocate exactly
    // as before.
    const Index recycle_cap = config.enable_recycling
        ? config.recycle_count * s_cap
        : 0;

    const Index capacity = recycle_cap + config.restart_blocks * s_cap;

    DenseBlock basis(n, capacity + 1);

    DenseMatrix H;
    DenseMatrix R(capacity + 1, Vector(capacity, 0.0));
    Vector g(capacity + 1, 0.0);
    Vector cosines(capacity, 0.0);
    Vector sines(capacity, 0.0);
    Index columns_rotated = 0;
    Index basis_cols = 0;

    const PartialCholeskyOptions partial_cholesky_options{
        config.partial_cholesky_stopping_rule,
        config.partial_cholesky_condition_limit
    };

    // Recycled-subspace seeding (GCRO-DR style). Let U = recycled_U and
    // C = orthonormalize(A U), giving A U = C T exactly (T upper triangular,
    // from CholQR's R). New blocks below orthogonalise against C (already
    // guaranteed by folding C into the leading columns of `basis`, which is
    // exactly what every block already orthogonalises against). Writing the
    // correction as U z_U + V z_V and substituting w := T z_U turns the
    // C-component of the residual into the trivial relation p - w (p = Cᵀr0),
    // decoupled from z_V; solving the combined least-squares system (seeded
    // with an identity block for w, see below) therefore gives w directly,
    // and z_U = T^{-1} w recovers the actual coefficients used in the final
    // solution update. k is the number of recycled columns actually used
    // this cycle (0 if recycling is off, recycled_U is empty, A U is
    // numerically rank deficient, or r_start already lies entirely in
    // span(A U)).
    Index k = 0;
    Scalar residual_before_block = beta;
    DenseBlock T; // A U = C T; needed after the loop to recover U's coefficients.

    if (config.enable_recycling && recycled_U.cols() > 0) {
        const Index requested_k = recycled_U.cols();

        DenseBlock AU(n, requested_k);
        for (Index i = 0; i < requested_k; ++i) {
            AU.set_column(i, A.multiply(recycled_U.get_column(i)));
        }

        const CholQRResult au_qr = cholqr(AU, partial_cholesky_options);
        const Index k_eff = au_qr.accepted_columns;

        if (k_eff > 0) {
            Vector p = transpose_multiply_vector(au_qr.Q, k_eff, r_start);

            Vector r0_perp = r_start;
            Vector negated_p = p;
            for (Scalar& value : negated_p) {
                value = -value;
            }
            multiply_add_columns(au_qr.Q, k_eff, negated_p, r0_perp);

            const Scalar beta_prime = norm2(r0_perp);

            if (beta_prime > 0.0) {
                for (Index i = 0; i < k_eff; ++i) {
                    basis.set_column(i, au_qr.Q.get_column(i));
                }

                scal(1.0 / beta_prime, r0_perp);
                basis.set_column(k_eff, r0_perp);
                basis_cols = k_eff + 1;

                // Seed columns represent the free substitution variable
                // w := T z_U (not U or C themselves), introduced so that the
                // combined least-squares system decouples cleanly: its own
                // coefficient is therefore the identity (w's contribution to
                // the residual along C is simply "-w"), with a zero row below
                // (no coupling into the new residual direction). z_U is
                // recovered from the solved w via T afterward.
                H.assign(k_eff + 1, Vector(k_eff, 0.0));
                for (Index col = 0; col < k_eff; ++col) {
                    H[col][col] = 1.0;
                }

                for (Index i = 0; i < k_eff; ++i) {
                    g[i] = p[i];
                }
                g[k_eff] = beta_prime;

                fold_hessenberg_columns(H, R, g, cosines, sines, columns_rotated, k_eff);

                result.residual_history.push_back(
                    {result.iterations, beta_prime, false, k_eff, true});

                residual_before_block = beta_prime;
                T = au_qr.R;
                k = k_eff;
            }
        }
    }

    if (k == 0) {
        H.assign(1, Vector());
        g[0] = beta;

        Vector q0 = r_start;
        scal(1.0 / beta, q0);
        basis.set_column(0, q0);
        basis_cols = 1;
    }

    // Adaptive s-step state. When config.adaptive_s is false, s_current stays
    // fixed at config.s_step and this loop behaves exactly as before. With
    // s_initial_probe the first block requests s_max, so the condition-limited
    // CholQR measures the widest stable width on the actual Krylov vectors;
    // the shrink-to-accepted rule below then turns that into the working width.
    auto clamp_s = [&](Index s) {
        return std::min(config.s_max, std::max(config.s_min, s));
    };
    Index s_current = config.adaptive_s ? clamp_s(config.s_step) : config.s_step;
    if (config.adaptive_s && config.s_initial_probe) {
        s_current = config.s_max;
    }
    Index consecutive_full = 0;

    // Candidates for the NEXT cycle's recycled subspace: only blocks
    // generated THIS cycle are eligible (never the recycled seed columns
    // themselves), scored by relative residual drop.
    std::vector<RecycleCandidate> candidates;

    // Newton/ScaledNewton bootstrap: no shifts exist until a cycle has run
    // once with Monomial and its Hessenberg has been mined for Ritz values.
    // shifts.empty() here means this IS that cycle - use Monomial regardless
    // of config.polynomial_basis, and extract bootstrap_shifts once the block
    // loop below finishes. On cycle 1 specifically (the only cycle where this
    // can trigger - see bootstrap_shifts below) recycled_U is necessarily
    // also still empty, so there is no risk of the recycling seed's identity
    // block contaminating the Ritz-value extraction.
    const bool bootstrapping =
        config.polynomial_basis != PolynomialBasisType::Monomial && shifts.empty();
    const PolynomialBasisType effective_basis_type =
        bootstrapping ? PolynomialBasisType::Monomial : config.polynomial_basis;

    for (Index block = 0; block < config.restart_blocks; ++block) {
        const Index requested = s_current;
        const Index block_col_start = basis_cols;

        SStepArnoldiResult block_result =
            sstep_arnoldi_block(A,
                                basis,
                                basis_cols,
                                requested,
                                effective_basis_type,
                                config.block_orthogonalization,
                                partial_cholesky_options,
                                shifts);

        if (block_result.accepted_columns > 0) {
            if (effective_basis_type == PolynomialBasisType::Monomial) {
                append_monomial_hessenberg_block(H,
                                                  block_result.R_old,
                                                  block_result.R_block);
            } else {
                append_shifted_hessenberg_block(H,
                                                block_result.R_old,
                                                block_result.R_block,
                                                block_result.used_shifts,
                                                block_result.used_scales);
            }

            for (Index col = 0; col < block_result.accepted_columns; ++col) {
                std::copy(block_result.Q_block.column(col),
                          block_result.Q_block.column(col) + n,
                          basis.column(basis_cols + col));
            }

            basis_cols += block_result.accepted_columns;
        }

        result.blocks_completed += 1;
        result.iterations += block_result.accepted_columns;

        // Fold the new Hessenberg columns into the rotated copy so the
        // least-squares residual estimate is available after every block.
        const Index total_columns = basis_cols - 1;
        fold_hessenberg_columns(H, R, g, cosines, sines, columns_rotated, total_columns);

        if (block_result.accepted_columns > 0) {
            const Scalar residual_estimate = std::abs(g[columns_rotated]);
            result.residual_history.push_back(
                {result.iterations, residual_estimate, false, requested});

            if (config.enable_recycling && config.recycle_count > 0
                && residual_before_block > 0.0) {
                const Scalar score =
                    (residual_before_block - residual_estimate) / residual_before_block;
                candidates.push_back(
                    {score, block_col_start, block_result.accepted_columns});
            }

            residual_before_block = residual_estimate;

            if (residual_estimate < config.tolerance) {
                result.converged = true;
                break;
            }
        }

        // A block that accepted no columns made no progress; the cycle cannot
        // continue regardless of the s-step policy.
        if (block_result.accepted_columns == 0) {
            break;
        }

        if (config.adaptive_s) {
            if (block_result.accepted_columns < requested) {
                // Truncated or condition-limited: shrink toward the accepted
                // width, then keep going with the remaining block budget.
                consecutive_full = 0;
                s_current = clamp_s(block_result.accepted_columns);
            } else {
                // Fully accepted: grow after enough consecutive stable blocks.
                ++consecutive_full;
                if (consecutive_full >= config.s_grow_after) {
                    s_current = clamp_s(s_current + 1);
                    consecutive_full = 0;
                }
            }
        } else if (block_result.truncated) {
            break;
        }
    }

    if (bootstrapping) {
        result.bootstrap_shifts = leja_order(compute_ritz_shifts(H));
    }

    if (!candidates.empty()) {
        std::sort(candidates.begin(), candidates.end(),
                  [](const RecycleCandidate& lhs, const RecycleCandidate& rhs) {
                      return lhs.score > rhs.score;
                  });

        const Index keep = std::min(config.recycle_count, candidates.size());
        Index total_cols = 0;
        for (Index i = 0; i < keep; ++i) {
            total_cols += candidates[i].col_count;
        }

        DenseBlock combined(n, total_cols);
        Index offset = 0;
        for (Index i = 0; i < keep; ++i) {
            for (Index col = 0; col < candidates[i].col_count; ++col) {
                std::copy(basis.column(candidates[i].col_start + col),
                          basis.column(candidates[i].col_start + col) + n,
                          combined.column(offset + col));
            }
            offset += candidates[i].col_count;
        }

        result.recycle_candidate_block = std::move(combined);
    }

    const Index inner_iterations = columns_rotated;

    if (inner_iterations == 0) {
        result.converged = false;
        return result;
    }

    Vector y = solve_upper_triangular(R, g, inner_iterations);

    if (k > 0) {
        const DenseBlock U_used = recycled_U.leading_columns(k);

        // The combined solve above returns w, the coefficient of C in the
        // augmented system; A U = C T means U's own coefficient is
        // z = T^{-1} w (see solve_upper_triangular_block).
        const Vector w_U(y.begin(), y.begin() + k);
        const Vector z_U = solve_upper_triangular_block(T, w_U);
        const Vector y_new(y.begin() + k, y.end());

        multiply_add_columns(U_used, k, z_U, result.x);
        multiply_add_columns_from(basis, k, inner_iterations - k, y_new, result.x);
    } else {
        multiply_add_columns(basis, inner_iterations, y, result.x);
    }

    return result;
}

} // namespace gmres
