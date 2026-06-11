#ifndef COMMUNICATION_AVOIDING_KRYLOV_BASIS_HPP
#define COMMUNICATION_AVOIDING_KRYLOV_BASIS_HPP

#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/polynomial_basis.hpp"

namespace gmres {

VectorList generate_krylov_block(const SparseMatrixCSR& A,
                                 const Vector& q,
                                 Index s,
                                 PolynomialBasisType basis_type);

}

#endif