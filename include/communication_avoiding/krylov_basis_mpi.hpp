#ifndef COMMUNICATION_AVOIDING_KRYLOV_BASIS_MPI_HPP
#define COMMUNICATION_AVOIDING_KRYLOV_BASIS_MPI_HPP

#include "common/types.hpp"
#include "communication_avoiding/polynomial_basis.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

DistributedVectorList generate_krylov_block_mpi(const DistributedSparseMatrixCSR& A,
                                                const DistributedVector& q,
                                                Index s,
                                                PolynomialBasisType basis_type);

}

#endif