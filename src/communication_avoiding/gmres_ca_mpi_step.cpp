#include "communication_avoiding/gmres_ca_mpi_step.hpp"

#include "common/dense_block.hpp"
#include "common/givens.hpp"
#include "communication_avoiding/hessenberg_assembly.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "communication_avoiding/sstep_arnoldi_mpi.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <algorithm>
#include <cmath>
#include <lapacke.h>
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace gmres {

namespace {

int mpi_count(Index count)
{
    if (count > static_cast<Index>(std::numeric_limits<int>::max())) {
        throw std::length_error("gmres_ca_mpi: reduction size exceeds the MPI count range.");
    }

    return static_cast<int>(count);
}

lapack_int lapack_dim(Index size)
{
    if (size > static_cast<Index>(std::numeric_limits<lapack_int>::max())) {
        throw std::length_error("gmres_ca_mpi_step: dimension exceeds the LAPACK range.");
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

// Deflated-cycle least-squares helpers. H, g and y are all replicated (H is
// assembled from the small replicated block factors; g is either [beta;0...] or
// an Allreduced projection), so these run identically on every rank with no
// communication. See gmres_ca_step.cpp for the rationale behind the SVD-based
// dgelsd and the finite-guarantee.
Vector solve_least_squares_dense(const DenseMatrix& H, const Vector& g)
{
    const Index rows = H.size();
    const Index cols = rows > 0 ? H.front().size() : 0;

    if (cols == 0) {
        return Vector();
    }

    const Index ldb = std::max(rows, cols);

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
    const double rcond = 1e-12;

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
        throw std::runtime_error("gmres_ca_mpi_step: LAPACKE_dgeqrf failed.");
    }

    info = LAPACKE_dorgqr(LAPACK_COL_MAJOR,
                          lapack_dim(rows),
                          lapack_dim(cols),
                          lapack_dim(std::min(rows, cols)),
                          a.data(),
                          lapack_dim(rows),
                          tau.data());
    if (info != 0) {
        throw std::runtime_error("gmres_ca_mpi_step: LAPACKE_dorgqr failed.");
    }

    DenseMatrix Q(rows, Vector(cols, 0.0));
    for (Index j = 0; j < cols; ++j) {
        for (Index i = 0; i < rows; ++i) {
            Q[i][j] = a[i + j * rows];
        }
    }

    return Q;
}

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

// 2-norm of a replicated vector (the deflated cycle's least-squares residual;
// no MPI reduction, every rank holds the same w).
Scalar euclidean_norm(const Vector& v)
{
    Scalar sum = 0.0;
    for (Scalar value : v) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

void validate_cycle_inputs(const DistributedSparseMatrixCSR& A,
                           const DistributedVector& b,
                           const DistributedVector& x_start,
                           const DistributedVector& r_start,
                           const GMRESConfig& config,
                           const char* who)
{
    const auto fail = [&](const char* what) {
        throw std::invalid_argument(std::string(who) + ": " + what);
    };

    if (A.global_rows() != A.global_cols()) {
        fail("requires a square matrix.");
    }
    if (b.global_size() != A.global_rows()) {
        fail("b has wrong global size.");
    }
    if (x_start.global_size() != A.global_cols()) {
        fail("x_start has wrong global size.");
    }
    if (r_start.global_size() != A.global_rows()) {
        fail("r_start has wrong global size.");
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

    check_compatible(b, r_start);
    check_compatible(b, x_start);
}

} // namespace

CAGMRESMPICycleResult gmres_ca_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                         const DistributedVector& b,
                                         const DistributedVector& x_start,
                                         const DistributedVector& r_start,
                                         Scalar beta,
                                         const GMRESConfig& config,
                                         const Vector& shifts)
{
    validate_cycle_inputs(A, b, x_start, r_start, config, "gmres_ca_mpi_cycle");

    CAGMRESMPICycleResult result;
    result.x = x_start;

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    const Index local_rows = r_start.local_size();
    MPI_Comm comm = r_start.communicator();

    const Index s_cap = config.adaptive_s
        ? std::max(config.s_step, config.s_max)
        : config.s_step;

    const Index capacity = config.restart_blocks * s_cap;

    DenseBlock local_basis(local_rows, capacity + 1);

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

    Vector q0 = r_start.local_values();
    for (Scalar& value : q0) {
        value /= beta;
    }
    local_basis.set_column(0, q0);
    basis_cols = 1;

    DistributedDenseBlock basis(r_start.global_size(),
                                r_start.local_start(),
                                std::move(local_basis),
                                comm);

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

        SStepArnoldiMPIResult block_result =
            sstep_arnoldi_block_mpi(A,
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
                          block_result.Q_block.column(col) + local_rows,
                          basis.local_block().column(basis_cols + col));
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
    multiply_add_columns(basis.local_block(), inner_iterations, y, result.x.local_values());

    return result;
}

CAGMRESMPICycleResult gmres_ca_dr_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                            const DistributedVector& b,
                                            const DistributedVector& x_start,
                                            const DistributedVector& r_start,
                                            Scalar beta,
                                            const GMRESConfig& config,
                                            const DistributedDeflationSubspace& deflation,
                                            const Vector& shifts)
{
    validate_cycle_inputs(A, b, x_start, r_start, config, "gmres_ca_dr_mpi_cycle");

    const Index k = deflation.k;
    if (k > 0) {
        const Index hbar_cols = deflation.Hbar.empty()
            ? 0
            : static_cast<Index>(deflation.Hbar.front().size());
        if (deflation.V.global_rows() != r_start.global_size()
            || deflation.V.local_start() != r_start.local_start()
            || deflation.V.local_rows() != r_start.local_size()
            || deflation.V.cols() != k + 1
            || static_cast<Index>(deflation.Hbar.size()) != k + 1
            || hbar_cols != k) {
            throw std::invalid_argument(
                "gmres_ca_dr_mpi_cycle: deflation subspace has an inconsistent shape.");
        }
    }

    CAGMRESMPICycleResult result;
    result.x = x_start;

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    const Index local_rows = r_start.local_size();
    MPI_Comm comm = r_start.communicator();

    const Index s_cap = config.adaptive_s
        ? std::max(config.s_step, config.s_max)
        : config.s_step;

    const Index capacity = (config.recycle_count + 1) + config.restart_blocks * s_cap;

    DenseBlock local_basis(local_rows, capacity + 1);

    const PartialCholeskyOptions partial_cholesky_options{
        config.partial_cholesky_stopping_rule,
        config.partial_cholesky_condition_limit
    };

    DenseMatrix H;
    Vector g;
    Index basis_cols = 0;
    bool deflated_start = false;

    if (k > 0) {
        for (Index i = 0; i <= k; ++i) {
            local_basis.set_column(i, deflation.V.local_block().get_column(i));
        }
    }

    DistributedDenseBlock basis(r_start.global_size(),
                                r_start.local_start(),
                                std::move(local_basis),
                                comm);

    if (k > 0) {
        H = deflation.Hbar;

        // g = V(:,0:k+1)^T r_start: local partial projections summed across
        // ranks (the only extra communication a deflated restart needs).
        g = transpose_multiply_vector(basis.local_block(), k + 1, r_start.local_values());
        if (!g.empty()) {
            MPI_Allreduce(MPI_IN_PLACE, g.data(), mpi_count(g.size()), MPI_DOUBLE, MPI_SUM, comm);
        }

        deflated_start = std::all_of(g.begin(), g.end(),
            [](Scalar value) { return std::isfinite(value); });

        if (deflated_start) {
            basis_cols = k + 1;
            result.residual_history.push_back(
                {result.iterations, beta, false, k, true});
        }
    }

    if (!deflated_start) {
        H.assign(1, Vector());
        g.assign(1, 0.0);
        g[0] = beta;

        Vector q0 = r_start.local_values();
        for (Scalar& value : q0) {
            value /= beta;
        }
        basis.local_block().set_column(0, q0);
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

        SStepArnoldiMPIResult block_result =
            sstep_arnoldi_block_mpi(A,
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
                          block_result.Q_block.column(col) + local_rows,
                          basis.local_block().column(basis_cols + col));
            }

            basis_cols += block_result.accepted_columns;
            g.resize(H.size(), 0.0);
        }

        result.blocks_completed += 1;
        result.iterations += block_result.accepted_columns;

        if (block_result.accepted_columns > 0) {
            const Vector y = solve_least_squares_dense(H, g);
            const Scalar residual_estimate = euclidean_norm(residual_coefficients(H, g, y));
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
    multiply_add_columns(basis.local_block(), total_cols, y, result.x.local_values());

    // GMRES-DR restart: next cycle's deflation subspace. H is replicated, so
    // the harmonic Ritz eigenproblem, QR and the small matmuls are identical on
    // every rank; only V_new = V_{m+1} q is distributed (a local combination of
    // this rank's basis columns), so no communication is needed here.
    if (config.recycle_count > 0) {
        const std::vector<Vector> ritz = harmonic_ritz_vectors(H, 0, config.recycle_count);
        const Index kk = static_cast<Index>(ritz.size());

        if (kk > 0 && kk <= total_cols) {
            const Index rows = H.size(); // total_cols + 1
            const Vector w = residual_coefficients(H, g, y);

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
                return result;
            }

            const DenseMatrix q = qr_orthonormal_columns(p_tilde); // rows x (kk+1)

            DenseBlock v_new_local(local_rows, kk + 1);
            for (Index c = 0; c < kk + 1; ++c) {
                Vector q_col(rows, 0.0);
                for (Index i = 0; i < rows; ++i) {
                    q_col[i] = q[i][c];
                }
                Vector column(local_rows, 0.0);
                multiply_add_columns(basis.local_block(), rows, q_col, column);
                v_new_local.set_column(c, column);
            }

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

            DistributedDenseBlock v_new(basis.global_rows(),
                                        basis.local_start(),
                                        std::move(v_new_local),
                                        comm);

            result.next_deflation =
                DistributedDeflationSubspace{std::move(v_new), std::move(hbar_new), kk};
        }
    }

    return result;
}

} // namespace gmres
