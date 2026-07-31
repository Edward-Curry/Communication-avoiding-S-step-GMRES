#ifndef COMMUNICATION_AVOIDING_GMRES_CA_STEP_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_STEP_HPP

#include "common/config.hpp"
#include "common/dense_block.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/ca_residual_history.hpp"

namespace gmres {

// A GMRES-DR deflation subspace carried across restart cycles. V holds k+1
// orthonormal columns and Hbar is (k+1) x k, together satisfying the Arnoldi
// relation A * V(:,0:k) = V(:,0:k+1) * Hbar exactly (inherited from the cycle
// that produced it - no matrix-vector products are needed to restart from it).
// The first k columns of V approximate the eigenvectors of the k smallest
// eigenvalues (harmonic Ritz vectors); the (k+1)-th column carries the residual
// direction so the next cycle's residual lies in span(V). Empty (k == 0) on the
// first cycle.
struct DeflationSubspace {
    DenseBlock V;      // n x (k+1), orthonormal columns
    DenseMatrix Hbar;  // (k+1) x k
    Index k = 0;
};

struct CAGMRESCycleResult {
    Vector x;
    CAResidualHistory residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;

    // The deflation subspace for the NEXT cycle, computed by GMRES-DR from this
    // cycle's harmonic Ritz vectors plus the residual direction (empty, k == 0,
    // when recycling is off or too few columns were built). Only gmres_ca_dr_
    // cycle populates this; the plain Givens gmres_ca_cycle leaves it empty.
    DeflationSubspace next_deflation;

    // Real, Leja-ordered Ritz-value shifts extracted from THIS cycle's own
    // Hessenberg matrix. Only populated when this cycle ran in bootstrap mode
    // (config.polynomial_basis wants Newton/ScaledNewton but the shifts
    // passed in were empty); empty otherwise, including on every later
    // cycle once shifts have been established (shifts are computed once,
    // not refreshed). Callers adopt this wholesale as the shift list for
    // subsequent cycles.
    Vector bootstrap_shifts;
};

// Plain restarted CA-GMRES cycle (no deflation): Arnoldi via s-step blocks,
// incremental Givens least-squares, warm restart. Used when
// config.enable_recycling is false. shifts (may be empty) supplies the
// Newton/ScaledNewton shift sequence established by an earlier cycle's
// bootstrap; empty means this cycle itself runs in bootstrap mode (uses
// Monomial regardless of config.polynomial_basis, and computes bootstrap_shifts
// from its own accumulated Hessenberg for the caller to adopt from then on).
CAGMRESCycleResult gmres_ca_cycle(const SparseMatrixCSR& A,
                                  const Vector& b,
                                  const Vector& x_start,
                                  const Vector& r_start,
                                  Scalar beta,
                                  const GMRESConfig& config,
                                  const Vector& shifts = Vector());

// GMRES-DR cycle (deflated restart): starts from the carried deflation subspace
// (empty on the first cycle), generates s-step Arnoldi blocks on top, solves the
// small dense least-squares problem (the leading deflation block of Hbar is
// full, not Hessenberg, so incremental Givens does not apply), and returns the
// next cycle's deflation subspace via result.next_deflation. Used when
// config.enable_recycling is true; config.recycle_count sets the number of
// harmonic Ritz vectors retained. shifts behaves as for gmres_ca_cycle.
CAGMRESCycleResult gmres_ca_dr_cycle(const SparseMatrixCSR& A,
                                     const Vector& b,
                                     const Vector& x_start,
                                     const Vector& r_start,
                                     Scalar beta,
                                     const GMRESConfig& config,
                                     const DeflationSubspace& deflation =
                                         DeflationSubspace(),
                                     const Vector& shifts = Vector());

}

#endif
