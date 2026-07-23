#include "communication_avoiding/krylov_basis_mpi.hpp"

#include "parallel/distributed_vector_ops.hpp"

#include <stdexcept>
#include <utility>

namespace gmres {

namespace {

DistributedVector apply_shifted_operator_mpi(const DistributedSparseMatrixCSR& A,
                                             const DistributedVector& q,
                                             Scalar shift)
{
    DistributedVector result = A.multiply(q);

    if (shift != 0.0) {
        axpy_local(-shift, q, result);
    }

    return result;
}

}

KrylovBlockMPIResult generate_krylov_block_mpi(const DistributedSparseMatrixCSR& A,
                                               const DistributedVector& q,
                                               Index s,
                                               PolynomialBasisType basis_type,
                                               const Vector& shifts)
{
    if (A.global_rows() != A.global_cols()) {
        throw std::invalid_argument("generate_krylov_block_mpi requires a square matrix.");
    }

    if (q.global_size() != A.global_cols()) {
        throw std::invalid_argument("generate_krylov_block_mpi: q has wrong global size.");
    }

    if (q.communicator() != A.communicator()) {
        throw std::invalid_argument("generate_krylov_block_mpi: communicator mismatch.");
    }

    if (s == 0) {
        throw std::invalid_argument("generate_krylov_block_mpi: s must be positive.");
    }

    if (basis_type != PolynomialBasisType::Monomial && shifts.empty()) {
        throw std::invalid_argument(
            "generate_krylov_block_mpi: Newton/ScaledNewton requires a nonempty shift list.");
    }

    KrylovBlockMPIResult result;
    DenseBlock local_block(q.local_size(), s);

    DistributedVector current = q;

    for (Index j = 0; j < s; ++j) {
        switch (basis_type) {
        case PolynomialBasisType::Monomial: {
            current = A.multiply(current);
            local_block.set_column(j, current.local_values());
            break;
        }
        case PolynomialBasisType::Newton: {
            current = apply_shifted_operator_mpi(A, current, shifts[j % shifts.size()]);
            local_block.set_column(j, current.local_values());
            break;
        }
        case PolynomialBasisType::ScaledNewton: {
            DistributedVector raw = apply_shifted_operator_mpi(A, current, shifts[j % shifts.size()]);
            const Scalar scale = norm2_mpi(raw);

            if (scale == 0.0) {
                throw std::runtime_error(
                    "generate_krylov_block_mpi: ScaledNewton produced an exact zero column.");
            }

            scal_local(1.0 / scale, raw);
            local_block.set_column(j, raw.local_values());
            result.column_scales.push_back(scale);
            current = raw;
            break;
        }
        }
    }

    result.block = DistributedDenseBlock(q.global_size(),
                                         q.local_start(),
                                         std::move(local_block),
                                         q.communicator());

    return result;
}

}
