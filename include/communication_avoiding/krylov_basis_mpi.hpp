#ifndef COMMUNICATION_AVOIDING_KRYLOV_BASIS_MPI_HPP
#define COMMUNICATION_AVOIDING_KRYLOV_BASIS_MPI_HPP

#include "common/types.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

struct KrylovBlockMPIResult {
    DistributedDenseBlock block;
    // Per-column rescale factors (ScaledNewton only); replicated (identical
    // on every rank, since each comes from norm2_mpi). Empty for
    // Monomial/Newton.
    Vector column_scales;
};

// Generates one s-step block starting from q. For Monomial, shifts is
// ignored. For Newton/ScaledNewton, shifts supplies the (real, Leja-ordered)
// shift sequence; if s exceeds shifts.size() the sequence is cycled from the
// start. Newton/ScaledNewton with an empty shifts is a caller error.
KrylovBlockMPIResult generate_krylov_block_mpi(const DistributedSparseMatrixCSR& A,
                                               const DistributedVector& q,
                                               Index s,
                                               PolynomialBasisType basis_type,
                                               const Vector& shifts = Vector());

}

#endif
