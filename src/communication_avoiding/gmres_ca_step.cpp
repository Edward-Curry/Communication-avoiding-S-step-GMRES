#include "communication_avoiding/gmres_ca_step.hpp"

#include "common/dense_block.hpp"
#include "common/givens.hpp"
#include "common/vector_ops.hpp"
#include "communication_avoiding/hessenberg_assembly.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "communication_avoiding/sstep_arnoldi.hpp"

#include <algorithm>
#include <cmath>
#include <lapacke.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gmres {

namespace {

lapack_int lapack_dim(Index size)
{
    if (size > static_cast<Index>(std::numeric_limits<lapack_int>::max())) {
        throw std::length_error("gmres_ca_step: dimension exceeds the LAPACK range.");
    }

    return static_cast<lapack_int>(size);
}

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

// Applies the accumulated Givens rotations to Hessenberg columns
// [columns_rotated, total_columns) of H, writing the rotated result into R
// and updating g and the rotation history in place. H, R and g are the
// least-squares state of the plain (non-deflated) cycle.
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

// Solves min_y ||g - H y||_2 for the tiny, replicated dense least-squares
// problem of a deflated (GMRES-DR) cycle, where H has a full leading block and
// incremental Givens does not apply. H is (rows x cols) with rows >= cols and
// g has length rows. Returns y (length cols).
//
// Uses the SVD-based dgelsd (minimum-norm solution) rather than QR-based dgels:
// the s-step monomial basis can make H numerically rank-deficient (the same
// ill-conditioning the block orthogonalization's condition limit guards
// against), and QR would divide by a ~zero pivot and return NaN. dgelsd
// truncates the tiny singular values and stays finite.
Vector solve_least_squares_dense(const DenseMatrix& H, const Vector& g)
{
    const Index rows = H.size();
    const Index cols = rows > 0 ? H.front().size() : 0;

    if (cols == 0) {
        return Vector();
    }

    const Index ldb = std::max(rows, cols);

    // Column-major copy for LAPACK.
    std::vector<double> a(rows * cols, 0.0);
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) {
            a[i + j * rows] = H[i][j];
        }
    }

    std::vector<double> rhs(ldb, 0.0);
    for (Index i = 0; i < rows; ++i) {
        rhs[i] = g[i];
    }

    std::vector<double> singular_values(std::min(rows, cols), 0.0);
    lapack_int rank = 0;
    const double rcond = 1e-12; // drop singular values below rcond * largest

    const lapack_int info = LAPACKE_dgelsd(LAPACK_COL_MAJOR,
                                           lapack_dim(rows),
                                           lapack_dim(cols),
                                           1,
                                           a.data(),
                                           lapack_dim(rows),
                                           rhs.data(),
                                           lapack_dim(ldb),
                                           singular_values.data(),
                                           rcond,
                                           &rank);

    // Never let a degenerate solve poison the iterate: a zero solution leaves x
    // unchanged (no progress) rather than propagating NaN/Inf into the residual.
    // The driver's per-cycle recompute + the deflation reset below recover from
    // it. This happens only on pathological (rank-collapsed) H; the SPD-like
    // problems that recycling targets stay well within dgelsd's comfort zone.
    Vector y(rhs.begin(), rhs.begin() + cols);
    bool y_finite = info == 0;
    for (Scalar value : y) {
        if (!std::isfinite(value)) {
            y_finite = false;
        }
    }
    if (!y_finite) {
        return Vector(cols, 0.0);
    }

    return y;
}

// Economy QR: returns Q (rows x cols, orthonormal columns) with the same span
// as M (rows x cols, rows >= cols). Used to orthonormalize the GMRES-DR restart
// basis [harmonic Ritz vectors | residual direction].
DenseMatrix qr_orthonormal_columns(const DenseMatrix& M)
{
    const Index rows = M.size();
    const Index cols = rows > 0 ? M.front().size() : 0;

    std::vector<double> a(rows * cols, 0.0);
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) {
            a[i + j * rows] = M[i][j];
        }
    }

    std::vector<double> tau(std::min(rows, cols), 0.0);

    lapack_int info = LAPACKE_dgeqrf(LAPACK_COL_MAJOR,
                                     lapack_dim(rows),
                                     lapack_dim(cols),
                                     a.data(),
                                     lapack_dim(rows),
                                     tau.data());
    if (info != 0) {
        throw std::runtime_error("gmres_ca_step: LAPACKE_dgeqrf failed.");
    }

    info = LAPACKE_dorgqr(LAPACK_COL_MAJOR,
                          lapack_dim(rows),
                          lapack_dim(cols),
                          lapack_dim(std::min(rows, cols)),
                          a.data(),
                          lapack_dim(rows),
                          tau.data());
    if (info != 0) {
        throw std::runtime_error("gmres_ca_step: LAPACKE_dorgqr failed.");
    }

    DenseMatrix Q(rows, Vector(cols, 0.0));
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) {
            Q[i][j] = a[i + j * rows];
        }
    }

    return Q;
}

// Residual coefficient vector w = g - H y (length H.size()); its norm is the
// least-squares residual and, in GMRES-DR, w is the extra restart direction.
Vector residual_coefficients(const DenseMatrix& H, const Vector& g, const Vector& y)
{
    const Index rows = H.size();
    const Index cols = rows > 0 ? H.front().size() : 0;

    Vector w(rows, 0.0);
    for (Index i = 0; i < rows; ++i) {
        Scalar acc = g[i];
        for (Index j = 0; j < cols; ++j) {
            acc -= H[i][j] * y[j];
        }
        w[i] = acc;
    }

    return w;
}

void validate_cycle_inputs(const SparseMatrixCSR& A,
                           const Vector& b,
                           const Vector& x_start,
                           const Vector& r_start,
                           const GMRESConfig& config,
                           const char* who)
{
    const auto fail = [&](const char* what) {
        throw std::invalid_argument(std::string(who) + ": " + what);
    };

    if (A.rows() != A.cols()) {
        fail("requires a square matrix.");
    }
    if (b.size() != A.rows()) {
        fail("b has wrong size.");
    }
    if (x_start.size() != A.cols()) {
        fail("x_start has wrong size.");
    }
    if (r_start.size() != A.rows()) {
        fail("r_start has wrong size.");
    }
    if (config.restart_blocks == 0) {
        fail("restart_blocks must be positive.");
    }
    if (config.s_step == 0) {
        fail("s_step must be positive.");
    }
    if (config.adaptive_s) {
        if (config.s_min == 0) {
            fail("s_min must be positive when adaptive_s is set.");
        }
        if (config.s_max < config.s_min) {
            fail("s_max must be >= s_min when adaptive_s is set.");
        }
    } else if (config.s_initial_probe) {
        fail("s_initial_probe requires adaptive_s.");
    }
}

} // namespace

CAGMRESCycleResult gmres_ca_cycle(const SparseMatrixCSR& A,
                                  const Vector& b,
                                  const Vector& x_start,
                                  const Vector& r_start,
                                  Scalar beta,
                                  const GMRESConfig& config,
                                  const Vector& shifts)
{
    validate_cycle_inputs(A, b, x_start, r_start, config, "gmres_ca_cycle");

    CAGMRESCycleResult result;
    result.x = x_start;

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    const Index n = A.rows();

    const Index s_cap = config.adaptive_s
        ? std::max(config.s_step, config.s_max)
        : config.s_step;

    const Index capacity = config.restart_blocks * s_cap;

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

    H.assign(1, Vector());
    g[0] = beta;

    Vector q0 = r_start;
    scal(1.0 / beta, q0);
    basis.set_column(0, q0);
    basis_cols = 1;

    auto clamp_s = [&](Index s) {
        return std::min(config.s_max, std::max(config.s_min, s));
    };
    Index s_current = config.adaptive_s ? clamp_s(config.s_step) : config.s_step;
    if (config.adaptive_s && config.s_initial_probe) {
        s_current = config.s_max;
    }
    Index consecutive_full = 0;

    const bool bootstrapping =
        config.polynomial_basis != PolynomialBasisType::Monomial && shifts.empty();
    const PolynomialBasisType effective_basis_type =
        bootstrapping ? PolynomialBasisType::Monomial : config.polynomial_basis;

    for (Index block = 0; block < config.restart_blocks; ++block) {
        const Index requested = s_current;

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

        const Index total_columns = basis_cols - 1;
        fold_hessenberg_columns(H, R, g, cosines, sines, columns_rotated, total_columns);

        if (block_result.accepted_columns > 0) {
            const Scalar residual_estimate = std::abs(g[columns_rotated]);
            result.residual_history.push_back(
                {result.iterations, residual_estimate, false, requested});

            if (residual_estimate < config.tolerance) {
                result.converged = true;
                break;
            }
        }

        if (block_result.accepted_columns == 0) {
            break;
        }

        if (config.adaptive_s) {
            if (block_result.accepted_columns < requested) {
                consecutive_full = 0;
                s_current = clamp_s(block_result.accepted_columns);
            } else {
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

    const Index inner_iterations = columns_rotated;

    if (inner_iterations == 0) {
        result.converged = false;
        return result;
    }

    Vector y = solve_upper_triangular(R, g, inner_iterations);
    multiply_add_columns(basis, inner_iterations, y, result.x);

    return result;
}

CAGMRESCycleResult gmres_ca_dr_cycle(const SparseMatrixCSR& A,
                                     const Vector& b,
                                     const Vector& x_start,
                                     const Vector& r_start,
                                     Scalar beta,
                                     const GMRESConfig& config,
                                     const DeflationSubspace& deflation,
                                     const Vector& shifts)
{
    validate_cycle_inputs(A, b, x_start, r_start, config, "gmres_ca_dr_cycle");

    const Index k = deflation.k;
    if (k > 0) {
        const Index hbar_cols = deflation.Hbar.empty()
            ? 0
            : static_cast<Index>(deflation.Hbar.front().size());
        if (deflation.V.rows() != A.rows()
            || deflation.V.cols() != k + 1
            || static_cast<Index>(deflation.Hbar.size()) != k + 1
            || hbar_cols != k) {
            throw std::invalid_argument(
                "gmres_ca_dr_cycle: deflation subspace has an inconsistent shape.");
        }
    }

    CAGMRESCycleResult result;
    result.x = x_start;

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    const Index n = A.rows();

    const Index s_cap = config.adaptive_s
        ? std::max(config.s_step, config.s_max)
        : config.s_step;

    // Deflated start uses up to recycle_count+1 seed columns; the block loop
    // adds restart_blocks worth of new columns on top.
    const Index capacity = (config.recycle_count + 1) + config.restart_blocks * s_cap;

    DenseBlock basis(n, capacity + 1);

    const PartialCholeskyOptions partial_cholesky_options{
        config.partial_cholesky_stopping_rule,
        config.partial_cholesky_condition_limit
    };

    // Least-squares state grows as blocks are appended: H is (basis_cols) x
    // (basis_cols-1), g has length basis_cols. In a deflated cycle the leading
    // k columns of H are full (from Hbar), so we re-solve densely each block.
    DenseMatrix H;
    Vector g;
    Index basis_cols = 0;

    bool deflated_start = false;

    if (k > 0) {
        for (Index i = 0; i <= k; ++i) {
            basis.set_column(i, deflation.V.get_column(i));
        }

        H = deflation.Hbar; // (k+1) x k, satisfies A V(:,0:k) = V(:,0:k+1) Hbar.

        // RHS = V(:,0:k+1)^T r_start. In exact arithmetic r_start lies in
        // span(V), so this recovers the residual coordinates; recomputing from
        // the actual residual keeps it honest against drift.
        g = transpose_multiply_vector(basis, k + 1, r_start);

        deflated_start = std::all_of(g.begin(), g.end(),
            [](Scalar value) { return std::isfinite(value); });

        if (deflated_start) {
            basis_cols = k + 1;
            result.residual_history.push_back(
                {result.iterations, beta, false, k, true});
        }
    }

    if (!deflated_start) {
        // No deflation, or a degraded subspace that produced a non-finite RHS:
        // fall back to a plain undeflated restart this cycle (and, via the empty
        // next_deflation, let the driver discard the bad subspace).
        H.assign(1, Vector());
        g.assign(1, 0.0);
        g[0] = beta;

        Vector q0 = r_start;
        scal(1.0 / beta, q0);
        basis.set_column(0, q0);
        basis_cols = 1;
    }

    auto clamp_s = [&](Index s) {
        return std::min(config.s_max, std::max(config.s_min, s));
    };
    Index s_current = config.adaptive_s ? clamp_s(config.s_step) : config.s_step;
    if (config.adaptive_s && config.s_initial_probe) {
        s_current = config.s_max;
    }
    Index consecutive_full = 0;

    const bool bootstrapping =
        config.polynomial_basis != PolynomialBasisType::Monomial && shifts.empty();
    const PolynomialBasisType effective_basis_type =
        bootstrapping ? PolynomialBasisType::Monomial : config.polynomial_basis;

    for (Index block = 0; block < config.restart_blocks; ++block) {
        const Index requested = s_current;

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
            g.resize(H.size(), 0.0); // new Arnoldi rows carry a zero RHS entry.
        }

        result.blocks_completed += 1;
        result.iterations += block_result.accepted_columns;

        if (block_result.accepted_columns > 0) {
            const Vector y = solve_least_squares_dense(H, g);
            const Scalar residual_estimate = norm2(residual_coefficients(H, g, y));
            result.residual_history.push_back(
                {result.iterations, residual_estimate, false, requested});

            if (residual_estimate < config.tolerance) {
                result.converged = true;
                break;
            }
        }

        if (block_result.accepted_columns == 0) {
            break;
        }

        if (config.adaptive_s) {
            if (block_result.accepted_columns < requested) {
                consecutive_full = 0;
                s_current = clamp_s(block_result.accepted_columns);
            } else {
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

    const Index total_cols = H.empty() ? 0 : H.front().size();

    if (total_cols == 0) {
        result.converged = false;
        return result;
    }

    const Vector y = solve_least_squares_dense(H, g);
    multiply_add_columns(basis, total_cols, y, result.x);

    // GMRES-DR restart: next cycle's deflation subspace from the k' smallest
    // harmonic Ritz vectors plus the residual direction. basis(:,0:total_cols+1)
    // is V_{m+1}; H is the (total_cols+1) x total_cols Hessenberg (full leading
    // block, Hessenberg tail).
    if (config.recycle_count > 0) {
        const std::vector<Vector> ritz = harmonic_ritz_vectors(H, 0, config.recycle_count);
        const Index kk = static_cast<Index>(ritz.size());

        if (kk > 0 && kk <= total_cols) {
            const Index rows = H.size(); // total_cols + 1
            const Vector w = residual_coefficients(H, g, y);

            // P~ = [ [g_1;0] ... [g_kk;0]  w ], (total_cols+1) x (kk+1).
            DenseMatrix p_tilde(rows, Vector(kk + 1, 0.0));
            for (Index c = 0; c < kk; ++c) {
                for (Index i = 0; i < total_cols; ++i) {
                    p_tilde[i][c] = ritz[c][i];
                }
            }
            for (Index i = 0; i < rows; ++i) {
                p_tilde[i][kk] = w[i];
            }

            bool p_tilde_finite = true;
            for (const Vector& row : p_tilde) {
                for (Scalar value : row) {
                    if (!std::isfinite(value)) {
                        p_tilde_finite = false;
                    }
                }
            }
            if (!p_tilde_finite) {
                // Numerically degenerate restart (e.g. a rank-collapsed s-step
                // basis): skip updating the deflation subspace this cycle and
                // keep the previous one rather than crash on a NaN restart.
                return result;
            }

            const DenseMatrix q = qr_orthonormal_columns(p_tilde); // rows x (kk+1)

            // V_new = V_{m+1} * q  -> n x (kk+1); basis has `rows` valid columns.
            DenseBlock v_new(n, kk + 1);
            for (Index c = 0; c < kk + 1; ++c) {
                Vector q_col(rows, 0.0);
                for (Index i = 0; i < rows; ++i) {
                    q_col[i] = q[i][c];
                }
                Vector column(n, 0.0);
                multiply_add_columns(basis, rows, q_col, column);
                v_new.set_column(c, column);
            }

            // Hbar_new = q^T H q_top, q_top = q(0:total_cols, 0:kk).
            DenseMatrix h_q(rows, Vector(kk, 0.0));
            for (Index i = 0; i < rows; ++i) {
                for (Index c = 0; c < kk; ++c) {
                    Scalar acc = 0.0;
                    for (Index j = 0; j < total_cols; ++j) {
                        acc += H[i][j] * q[j][c];
                    }
                    h_q[i][c] = acc;
                }
            }

            DenseMatrix hbar_new(kk + 1, Vector(kk, 0.0));
            for (Index a = 0; a < kk + 1; ++a) {
                for (Index c = 0; c < kk; ++c) {
                    Scalar acc = 0.0;
                    for (Index i = 0; i < rows; ++i) {
                        acc += q[i][a] * h_q[i][c];
                    }
                    hbar_new[a][c] = acc;
                }
            }

            result.next_deflation =
                DeflationSubspace{std::move(v_new), std::move(hbar_new), kk};
        }
    }

    return result;
}

} // namespace gmres
