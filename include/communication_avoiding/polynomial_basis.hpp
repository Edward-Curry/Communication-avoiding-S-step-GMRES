#ifndef COMMUNICATION_AVOIDING_POLYNOMIAL_BASIS_HPP
#define COMMUNICATION_AVOIDING_POLYNOMIAL_BASIS_HPP

#include "common/types.hpp"

#include <vector>

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

// Computes the smallest-magnitude harmonic Ritz vectors of the trailing
// Arnoldi block of an accumulated (m+1) x m Hessenberg, for GCRO-DR-style
// subspace recycling: these approximate the eigenvectors belonging to the
// smallest eigenvalues, i.e. the directions GMRES resolves slowest and that
// recur cycle after cycle (the ones worth carrying across restarts).
//
// leading_offset skips the first columns of H (the recycled-subspace seed
// columns, whose Hessenberg entries are the identity block, not Arnoldi
// data); the trailing m' = m - leading_offset columns are the genuine
// Arnoldi Hessenberg of the (possibly deflated) operator, and harmonic Ritz
// is taken over exactly that clean block. Pass leading_offset = 0 for an
// unseeded cycle.
//
// Returns up to num_wanted coefficient vectors, each of length m'; column c
// of the recycled subspace is basis(:, leading_offset : m) * result[c]. A
// complex-conjugate eigenpair contributes its eigenvector's real and
// imaginary parts as two real vectors (both spanning the same 2D invariant
// subspace), never exceeding num_wanted in total. Solves the harmonic
// eigenproblem M g = theta g with M = H_sq + h_last^2 (H_sq^-T e_m) e_m^T;
// falls back to ordinary Ritz vectors (eigenvectors of H_sq) if H_sq is
// numerically singular. H is replicated, so this is a purely local
// computation identical on every MPI rank.
std::vector<Vector> harmonic_ritz_vectors(const DenseMatrix& hessenberg,
                                          Index leading_offset,
                                          Index num_wanted);

}

#endif
