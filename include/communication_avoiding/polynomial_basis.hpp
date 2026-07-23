#ifndef COMMUNICATION_AVOIDING_POLYNOMIAL_BASIS_HPP
#define COMMUNICATION_AVOIDING_POLYNOMIAL_BASIS_HPP

#include "common/types.hpp"

namespace gmres {

enum class PolynomialBasisType {
    Monomial,
    Newton,
    ScaledNewton
};

// Estimates eigenvalues (Ritz values) of A from the square leading m x m
// block of an accumulated (m+1) x m Hessenberg matrix, via LAPACKE_dgeev.
// Returns only the (numerically) real ones - a complex pair with |imag| not
// negligible relative to |real| is discarded rather than approximated
// wrongly, since this project only supports real Newton shifts for now. Can
// return fewer values than m, or none at all for a fully complex spectrum.
Vector compute_ritz_shifts(const DenseMatrix& hessenberg);

// Greedily reorders shifts by the classical Leja criterion: pick the
// largest-magnitude shift first, then repeatedly pick whichever remaining
// shift maximizes the product of distances to those already chosen. This is
// what keeps the resulting Newton polynomial numerically stable to evaluate
// in sequence - using the shifts in an arbitrary order is not safe.
Vector leja_order(const Vector& shifts);

}

#endif
