/**
 * @file include/communication_avoiding/polynomial_basis.hpp
 * @brief Declares polynomial-basis and recycling utilities.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_POLYNOMIAL_BASIS_HPP
#define COMMUNICATION_AVOIDING_POLYNOMIAL_BASIS_HPP

#include "common/types.hpp"

#include <vector>

namespace gmres {

/**
 * @brief Selects the polynomial basis for an s-step block.
 */
enum class PolynomialBasisType {
    Monomial,
    Newton,
    ScaledNewton
};

/**
 * @brief Extracts numerically real Ritz values from a Hessenberg matrix.
 * @param hessenberg Accumulated Arnoldi Hessenberg matrix.
 * @return Real Ritz values suitable as Newton shifts.
 */
Vector compute_ritz_shifts(const DenseMatrix& hessenberg);

/**
 * @brief Reorders shifts using the Leja criterion.
 * @param shifts Unordered real shifts.
 * @return Leja-ordered shifts.
 */
Vector leja_order(const Vector& shifts);

/**
 * @brief Computes harmonic-Ritz coefficients for recycling.
 * @param hessenberg Accumulated Arnoldi Hessenberg matrix.
 * @param leading_offset Number of recycled seed columns to skip.
 * @param num_wanted Maximum number of coefficient vectors.
 * @return Harmonic-Ritz coefficient vectors for the active Arnoldi block.
 */
std::vector<Vector> harmonic_ritz_vectors(const DenseMatrix& hessenberg,
                                          Index leading_offset,
                                          Index num_wanted);

}

#endif
