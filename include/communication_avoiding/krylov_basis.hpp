#ifndef COMMUNICATION_AVOIDING_KRYLOV_BASIS_HPP
#define COMMUNICATION_AVOIDING_KRYLOV_BASIS_HPP

#include "common/dense_block.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/polynomial_basis.hpp"

namespace gmres {

struct KrylovBlockResult {
    DenseBlock block;
    // Per-column rescale factors applied during generation (ScaledNewton
    // only - each column is divided by its own norm right after being
    // generated, and by the PREVIOUS column's scaled value, to keep the raw
    // vectors from growing unboundedly across many shifted products). Empty
    // for Monomial/Newton, where generation applies no rescaling.
    Vector column_scales;
};

// Generates one s-step block starting from q. For Monomial, shifts is
// ignored. For Newton/ScaledNewton, shifts supplies the (real, Leja-ordered)
// shift sequence; if s exceeds shifts.size() the sequence is cycled from the
// start. Newton/ScaledNewton with an empty shifts is a caller error (the
// cycle driving this should fall back to Monomial when no shifts are
// available yet).
KrylovBlockResult generate_krylov_block(const SparseMatrixCSR& A,
                                        const Vector& q,
                                        Index s,
                                        PolynomialBasisType basis_type,
                                        const Vector& shifts = Vector());

}

#endif
