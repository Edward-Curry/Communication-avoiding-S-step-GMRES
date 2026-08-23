/**
 * @file src/communication_avoiding/krylov_basis.cpp
 * @brief Implements sequential s-step Krylov block generation.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "communication_avoiding/krylov_basis.hpp"

#include "common/vector_ops.hpp"

#include <stdexcept>

namespace gmres {

namespace {

Vector apply_shifted_operator(const SparseMatrixCSR& A, const Vector& q, Scalar shift)
{
    Vector result = A.multiply(q);

    if (shift != 0.0) {
        axpy(-shift, q, result);
    }

    return result;
}

}

KrylovBlockResult generate_krylov_block(const SparseMatrixCSR& A,
                                        const Vector& q,
                                        Index s,
                                        PolynomialBasisType basis_type,
                                        const Vector& shifts)
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

    if (basis_type != PolynomialBasisType::Monomial && shifts.empty()) {
        throw std::invalid_argument(
            "generate_krylov_block: Newton/ScaledNewton requires a nonempty shift list.");
    }

    KrylovBlockResult result;
    result.block = DenseBlock(q.size(), s);

    Vector current = q;

    for (Index j = 0; j < s; ++j) {
        switch (basis_type) {
        case PolynomialBasisType::Monomial: {
            current = A.multiply(current);
            result.block.set_column(j, current);
            break;
        }
        case PolynomialBasisType::Newton: {
            current = apply_shifted_operator(A, current, shifts[j % shifts.size()]);
            result.block.set_column(j, current);
            break;
        }
        case PolynomialBasisType::ScaledNewton: {
            Vector raw = apply_shifted_operator(A, current, shifts[j % shifts.size()]);
            const Scalar scale = norm2(raw);

            if (scale == 0.0) {
                throw std::runtime_error(
                    "generate_krylov_block: ScaledNewton produced an exact zero column.");
            }

            scal(1.0 / scale, raw);
            result.block.set_column(j, raw);
            result.column_scales.push_back(scale);
            current = raw;
            break;
        }
        }
    }

    return result;
}

}
