#include "communication_avoiding/gmres_ca_mpi_step.hpp"

#include "common/givens.hpp"
#include "communication_avoiding/hessenberg_assembly.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "communication_avoiding/sstep_arnoldi_mpi.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <cmath>
#include <stdexcept>

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

void update_solution_mpi(DistributedVector& x,
                         const DistributedVectorList& basis,
                         const Vector& y)
{
    if (basis.size() < y.size()) {
        throw std::invalid_argument("update_solution_mpi: not enough basis vectors.");
    }

    for (Index j = 0; j < y.size(); ++j) {
        axpy_local(y[j], basis[j], x);
    }
}

void apply_givens_to_least_squares(DenseMatrix& H, Vector& g, Index cols)
{
    for (Index j = 0; j < cols; ++j) {
        Scalar c = 0.0;
        Scalar s = 0.0;

        generate_givens(H[j][j], H[j + 1][j], c, s);

        for (Index col = j; col < cols; ++col) {
            apply_givens(c, s, H[j][col], H[j + 1][col]);
        }

        apply_givens(c, s, g[j], g[j + 1]);
    }
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

    check_compatible(b, r_start);
    check_compatible(b, x_start);

    CAGMRESMPICycleResult result;
    result.x = x_start;

    if (beta == 0.0) {
        result.converged = true;
        return result;
    }

    DistributedVectorList basis;

    DistributedVector q0 = r_start;
    scal_local(1.0 / beta, q0);
    basis.push_back(q0);

    DenseMatrix H(1);

    for (Index block = 0; block < config.restart_blocks; ++block) {
        SStepArnoldiMPIResult block_result =
            sstep_arnoldi_block_mpi(A,
                                    basis,
                                    config.s_step,
                                    PolynomialBasisType::Monomial,
                                    config.block_orthogonalization);

        if (block_result.accepted_columns > 0) {
            append_monomial_hessenberg_block(H,
                                              block_result.R_old,
                                              block_result.R_block);
        }

        for (const DistributedVector& q : block_result.Q_block) {
            basis.push_back(q);
        }

        result.blocks_completed += 1;
        result.iterations += block_result.accepted_columns;

        if (block_result.truncated) {
            break;
        }
    }

    const Index inner_iterations = basis.size() - 1;

    if (inner_iterations == 0) {
        result.converged = false;
        return result;
    }

    Vector g(inner_iterations + 1, 0.0);
    g[0] = beta;

    apply_givens_to_least_squares(H, g, inner_iterations);

    const Scalar residual_estimate = std::abs(g[inner_iterations]);
    result.residual_history.push_back(residual_estimate);

    Vector y = solve_upper_triangular(H, g, inner_iterations);

    update_solution_mpi(result.x, basis, y);

    result.converged = residual_estimate < config.tolerance;

    return result;
}

} // namespace gmres
