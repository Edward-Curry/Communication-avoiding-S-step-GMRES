#include "communication_avoiding/gmres_ca_mpi_step.hpp"

#include "common/dense_block.hpp"
#include "common/givens.hpp"
#include "communication_avoiding/cholqr_mpi.hpp"
#include "communication_avoiding/hessenberg_assembly.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "communication_avoiding/sstep_arnoldi_mpi.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mpi.h>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gmres {

namespace {

int mpi_count(Index count)
{
    if (count > static_cast<Index>(std::numeric_limits<int>::max())) {
        throw std::length_error("gmres_ca_mpi_cycle: reduction size exceeds the MPI count range.");
    }

    return static_cast<int>(count);
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

// Solves the small k x k upper triangular system T z = w. Used to recover the
// recycled-subspace coefficients: the combined least-squares solve below
// returns w (the coefficient of C = orthonormalize(A U) in the augmented
// system), and since A U = C T, the coefficient of U itself is z = T^{-1} w.
// T is replicated on every rank, so this is a purely local computation.
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
// generated - the same incremental Givens-QR process either way. H, R and g
// are replicated on every rank, so this involves no communication.
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

CAGMRESMPICycleResult gmres_ca_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                         const DistributedVector& b,
                                         const DistributedVector& x_start,
                                         const DistributedVector& r_start,
                                         Scalar beta,
                                         const GMRESConfig& config,
                                         const DistributedDenseBlock& recycled_U)
{
    if (A.global_rows() != A.global_cols()) {
        throw std::invalid_argument("gmres_ca_mpi_cycle requires a square matrix.");
    }

    if (b.global_size() != A.global_rows()) {
        throw std::invalid_argument("gmres_ca_mpi_cycle: b has wrong global size.");
    }

    if (x_start.global_size() != A.global_cols()) {
        throw std::invalid_argument("gmres_ca_mpi_cycle: x_start has wrong global size.");
    }

    if (r_start.global_size() != A.global_rows()) {
        throw std::invalid_argument("gmres_ca_mpi_cycle: r_start has wrong global size.");
    }

    if (config.restart_blocks == 0) {
        throw std::invalid_argument("gmres_ca_mpi_cycle: restart_blocks must be positive.");
    }

    if (config.s_step == 0) {
        throw std::invalid_argument("gmres_ca_mpi_cycle: s_step must be positive.");
    }

    if (config.adaptive_s) {
        if (config.s_min == 0) {
            throw std::invalid_argument("gmres_ca_mpi_cycle: s_min must be positive when adaptive_s is set.");
        }

        if (config.s_max < config.s_min) {
            throw std::invalid_argument("gmres_ca_mpi_cycle: s_max must be >= s_min when adaptive_s is set.");
        }
    } else if (config.s_initial_probe) {
        throw std::invalid_argument("gmres_ca_mpi_cycle: s_initial_probe requires adaptive_s.");
    }

    if (recycled_U.cols() > 0
        && (recycled_U.global_rows() != r_start.global_size()
            || recycled_U.local_start() != r_start.local_start()
            || recycled_U.local_rows() != r_start.local_size()
            || recycled_U.communicator() != r_start.communicator())) {
        throw std::invalid_argument("gmres_ca_mpi_cycle: recycled_U layout does not match r_start.");
    }

    check_compatible(b, r_start);
    check_compatible(b, x_start);

    CAGMRESMPICycleResult result;
    result.x = x_start;

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    const Index local_rows = r_start.local_size();
    MPI_Comm comm = r_start.communicator();

    // Worst-case block width per block. Adaptive s can grow up to s_max, so the
    // basis and least-squares buffers must be sized for that upper bound. Every
    // rank derives this from the same replicated config, so all ranks agree.
    const Index s_cap = config.adaptive_s
        ? std::max(config.s_step, config.s_max)
        : config.s_step;

    // Worst-case recycled-subspace width; zero when recycling is off, so
    // disabled runs allocate exactly as before.
    const Index recycle_cap = config.enable_recycling
        ? config.recycle_count * s_cap
        : 0;

    const Index capacity = recycle_cap + config.restart_blocks * s_cap;

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

    // Recycled-subspace seeding (GCRO-DR style); see gmres_ca_step.cpp for the
    // full derivation. Every quantity that feeds the shared decision below
    // (k, beta_prime, T, p) is either already replicated (config) or made
    // replicated by an explicit Allreduce, so all ranks agree.
    Index k = 0;
    Scalar residual_before_block = beta;
    DenseBlock T; // A U = C T; needed after the loop to recover U's coefficients.

    if (config.enable_recycling && recycled_U.cols() > 0) {
        const Index requested_k = recycled_U.cols();

        DenseBlock AU_local(local_rows, requested_k);
        for (Index i = 0; i < requested_k; ++i) {
            const DistributedVector u_i(recycled_U.global_rows(),
                                        recycled_U.local_start(),
                                        recycled_U.local_block().get_column(i),
                                        recycled_U.communicator());
            const DistributedVector Au_i = A.multiply(u_i);
            AU_local.set_column(i, Au_i.local_values());
        }

        DistributedDenseBlock AU(recycled_U.global_rows(),
                                 recycled_U.local_start(),
                                 std::move(AU_local),
                                 recycled_U.communicator());

        const CholQRMPIResult au_qr = cholqr_mpi(AU, partial_cholesky_options);
        const Index k_eff = au_qr.accepted_columns;

        if (k_eff > 0) {
            Vector p = transpose_multiply_vector(au_qr.Q.local_block(), k_eff, r_start.local_values());

            if (!p.empty()) {
                MPI_Allreduce(MPI_IN_PLACE, p.data(), mpi_count(p.size()), MPI_DOUBLE, MPI_SUM, comm);
            }

            Vector r0_perp_local = r_start.local_values();
            Vector negated_p = p;
            for (Scalar& value : negated_p) {
                value = -value;
            }
            multiply_add_columns(au_qr.Q.local_block(), k_eff, negated_p, r0_perp_local);

            const DistributedVector r0_perp(r_start.global_size(),
                                            r_start.local_start(),
                                            r0_perp_local,
                                            comm);
            const Scalar beta_prime = norm2_mpi(r0_perp);

            if (beta_prime > 0.0) {
                for (Index i = 0; i < k_eff; ++i) {
                    local_basis.set_column(i, au_qr.Q.local_block().get_column(i));
                }

                for (Scalar& value : r0_perp_local) {
                    value /= beta_prime;
                }
                local_basis.set_column(k_eff, r0_perp_local);
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

        Vector q0 = r_start.local_values();
        for (Scalar& value : q0) {
            value /= beta;
        }
        local_basis.set_column(0, q0);
        basis_cols = 1;
    }

    DistributedDenseBlock basis(r_start.global_size(),
                                r_start.local_start(),
                                std::move(local_basis),
                                comm);

    // Adaptive s-step state. When config.adaptive_s is false, s_current stays
    // fixed at config.s_step and this loop behaves exactly as before. The
    // policy reads only replicated config and the accepted_columns count (which
    // comes from an Allreduce), so every rank chooses the same width.
    auto clamp_s = [&](Index s) {
        return std::min(config.s_max, std::max(config.s_min, s));
    };
    Index s_current = config.adaptive_s ? clamp_s(config.s_step) : config.s_step;
    if (config.adaptive_s && config.s_initial_probe) {
        // Probe: the first block requests s_max; the condition-limited CholQR
        // (replicated result of the Gram Allreduce) picks the stable width, so
        // every rank derives the same working width.
        s_current = config.s_max;
    }
    Index consecutive_full = 0;

    // Candidates for the NEXT cycle's recycled subspace: only blocks
    // generated THIS cycle are eligible, scored by relative residual drop.
    // Every rank computes the same list from replicated data.
    std::vector<RecycleCandidate> candidates;

    for (Index block = 0; block < config.restart_blocks; ++block) {
        const Index requested = s_current;
        const Index block_col_start = basis_cols;

        SStepArnoldiMPIResult block_result =
            sstep_arnoldi_block_mpi(A,
                                    basis,
                                    basis_cols,
                                    requested,
                                    PolynomialBasisType::Monomial,
                                    config.block_orthogonalization,
                                    partial_cholesky_options);

        if (block_result.accepted_columns > 0) {
            append_monomial_hessenberg_block(H,
                                              block_result.R_old,
                                              block_result.R_block);

            for (Index col = 0; col < block_result.accepted_columns; ++col) {
                std::copy(block_result.Q_block.column(col),
                          block_result.Q_block.column(col) + local_rows,
                          basis.local_block().column(basis_cols + col));
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

        DenseBlock combined_local(local_rows, total_cols);
        Index offset = 0;
        for (Index i = 0; i < keep; ++i) {
            for (Index col = 0; col < candidates[i].col_count; ++col) {
                std::copy(basis.local_block().column(candidates[i].col_start + col),
                          basis.local_block().column(candidates[i].col_start + col) + local_rows,
                          combined_local.column(offset + col));
            }
            offset += candidates[i].col_count;
        }

        result.recycle_candidate_block = DistributedDenseBlock(
            basis.global_rows(), basis.local_start(), std::move(combined_local), comm);
    }

    const Index inner_iterations = columns_rotated;

    if (inner_iterations == 0) {
        result.converged = false;
        return result;
    }

    Vector y = solve_upper_triangular(R, g, inner_iterations);

    if (k > 0) {
        const DistributedDenseBlock U_used = recycled_U.leading_columns(k);

        // The combined solve above returns w, the coefficient of C in the
        // augmented system; A U = C T means U's own coefficient is
        // z = T^{-1} w (see solve_upper_triangular_block).
        const Vector w_U(y.begin(), y.begin() + k);
        const Vector z_U = solve_upper_triangular_block(T, w_U);
        const Vector y_new(y.begin() + k, y.end());

        multiply_add_columns(U_used.local_block(), k, z_U, result.x.local_values());
        multiply_add_columns_from(basis.local_block(), k, inner_iterations - k, y_new, result.x.local_values());
    } else {
        multiply_add_columns(basis.local_block(), inner_iterations, y, result.x.local_values());
    }

    return result;
}

} // namespace gmres
