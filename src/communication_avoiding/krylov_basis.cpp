#include "communication_avoiding/krylov_basis.hpp"

#include <stdexcept>

namespace gmres {

DenseBlock generate_krylov_block(const SparseMatrixCSR& A,
                                 const Vector& q,
                                 Index s,
                                 PolynomialBasisType basis_type)
{
    if (A.rows() != A.cols()) {
        throw std::invalid_argument("generate_krylov_block requires a square matrix.");
    }

    if (q.size() != A.cols()) {
        throw std::invalid_argument("generate_krylov_block: q has wrong size.");
    }

    if (s == 0) {
        throw std::invalid_argument("generate_krylov_block: s must be positive.");
    }

    if (basis_type != PolynomialBasisType::Monomial) {
        throw std::invalid_argument("Only Monomial basis is implemented for now.");
    }

    DenseBlock block(q.size(), s);

    Vector current = q;

    for (Index j = 0; j < s; ++j) {
        current = A.multiply(current);
        block.set_column(j, current);
    }

    return block;
}

}
