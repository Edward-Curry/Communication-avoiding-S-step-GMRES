/**
 * @file include/communication_avoiding/krylov_basis.hpp
 * @brief Declares sequential s-step Krylov-block generation.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_KRYLOV_BASIS_HPP
#define COMMUNICATION_AVOIDING_KRYLOV_BASIS_HPP

#include "common/dense_block.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/polynomial_basis.hpp"

namespace gmres {

/**
 * @brief Stores one generated sequential Krylov block.
 */
struct KrylovBlockResult {
    DenseBlock block;
    /// @brief Per-column scales used by the Scaled-Newton basis.
    Vector column_scales;
};

/**
 * @brief Generates one sequential s-step Krylov block.
 * @param A System matrix.
 * @param q Seed vector.
 * @param s Requested block width.
 * @param basis_type Polynomial basis type.
 * @param shifts Leja-ordered Newton shifts when required.
 * @return Generated block and any Scaled-Newton factors.
 */
KrylovBlockResult generate_krylov_block(const SparseMatrixCSR& A,
                                        const Vector& q,
                                        Index s,
                                        PolynomialBasisType basis_type,
                                        const Vector& shifts = Vector());

}

#endif
