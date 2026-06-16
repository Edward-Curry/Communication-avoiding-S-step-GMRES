#include "communication_avoiding/krylov_basis_mpi.hpp"

#include <stdexcept>

namespace gmres {

DistributedVectorList generate_krylov_block_mpi(const DistributedSparseMatrixCSR& A,
                                                const DistributedVector& q,
                                                Index s,
                                                PolynomialBasisType basis_type)
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

    if (basis_type != PolynomialBasisType::Monomial) {
        throw std::invalid_argument("Only Monomial basis is implemented for MPI CA Krylov blocks for now.");
    }

    DistributedVectorList block;
    block.reserve(s);

    DistributedVector current = q;

    for (Index j = 0; j < s; ++j) {
        current = A.multiply(current);
        block.push_back(current);
    }

    return block;
}

}