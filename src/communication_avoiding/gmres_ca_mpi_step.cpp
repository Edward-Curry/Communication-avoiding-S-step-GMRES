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
#include <stdexcept>
#include <utility>

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

} // namespace

CAGMRESMPICycleResult gmres_ca_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                         const DistributedVector& b,
                                         const DistributedVector& x_start,
                                         const DistributedVector& r_start,
                                         Scalar beta,
                                         const GMRESConfig& config)
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

    check_compatible(b, r_start);
    check_compatible(b, x_start);

    CAGMRESMPICycleResult result;
    result.x = x_start;

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    const Index local_rows = r_start.local_size();

    // Worst-case block width per block. Adaptive s can grow up to s_max, so the
    // basis and least-squares buffers must be sized for that upper bound. Every
    // rank derives this from the same replicated config, so all ranks agree.
    const Index s_cap = config.adaptive_s
        ? std::max(config.s_step, config.s_max)
        : config.s_step;
    const Index capacity = config.restart_blocks * s_cap;

    DenseBlock local_basis(local_rows, capacity + 1);

    Vector q0 = r_start.local_values();
    for (Scalar& value : q0) {
        value /= beta;
    }
    local_basis.set_column(0, q0);

    DistributedDenseBlock basis(r_start.global_size(),
                                r_start.local_start(),
                                std::move(local_basis),
                                r_start.communicator());
    Index basis_cols = 1;

    DenseMatrix H(1);

    // Incrementally rotated copy of the Hessenberg for the least-squares
    // problem. H itself must stay untransformed because later Hessenberg
    // assembly steps read its raw columns. H, R and g are replicated on
    // every rank, so this involves no communication.
    DenseMatrix R(capacity + 1, Vector(capacity, 0.0));
    Vector g(capacity + 1, 0.0);
    g[0] = beta;

    Vector cosines(capacity, 0.0);
    Vector sines(capacity, 0.0);
    Index columns_rotated = 0;

    const PartialCholeskyOptions partial_cholesky_options{
        config.partial_cholesky_stopping_rule,
        config.partial_cholesky_condition_limit
    };

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

    for (Index block = 0; block < config.restart_blocks; ++block) {
        const Index requested = s_current;

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

        if (block_result.accepted_columns > 0) {
            const Scalar residual_estimate = std::abs(g[columns_rotated]);
            result.residual_history.push_back(
                {result.iterations, residual_estimate, false, requested});

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

    const Index inner_iterations = columns_rotated;

    if (inner_iterations == 0) {
        result.converged = false;
        return result;
    }

    Vector y = solve_upper_triangular(R, g, inner_iterations);

    multiply_add_columns(basis.local_block(),
                         inner_iterations,
                         y,
                         result.x.local_values());

    return result;
}

} // namespace gmres
