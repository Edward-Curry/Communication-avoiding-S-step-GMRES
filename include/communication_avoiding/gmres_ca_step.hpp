#ifndef COMMUNICATION_AVOIDING_GMRES_CA_STEP_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_STEP_HPP

#include "common/config.hpp"
#include "common/dense_block.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/ca_residual_history.hpp"

namespace gmres {

struct CAGMRESCycleResult {
    Vector x;
    CAResidualHistory residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;

    // The recycle_count newly generated blocks with the largest relative
    // residual drop this cycle, ranked best-first and concatenated into one
    // block (empty, 0 columns, when config.enable_recycling is false or no
    // block made progress). Callers adopt this wholesale as the next cycle's
    // recycled subspace.
    DenseBlock recycle_candidate_block;

    // Real, Leja-ordered Ritz-value shifts extracted from THIS cycle's own
    // Hessenberg matrix. Only populated when this cycle ran in bootstrap mode
    // (config.polynomial_basis wants Newton/ScaledNewton but the shifts
    // passed in were empty); empty otherwise, including on every later
    // cycle once shifts have been established (shifts are computed once,
    // not refreshed). Callers adopt this wholesale as the shift list for
    // subsequent cycles.
    Vector bootstrap_shifts;
};

// recycled_U (may have 0 columns) supplies a subspace from a previous cycle
// to augment the start of this cycle's search space; see GMRESConfig::
// enable_recycling for the construction and cost. shifts (may be empty)
// supplies the Newton/ScaledNewton shift sequence established by an earlier
// cycle's bootstrap; empty means this cycle itself runs in bootstrap mode
// (uses Monomial regardless of config.polynomial_basis, and computes
// bootstrap_shifts from its own accumulated Hessenberg for the caller to
// adopt from then on).
CAGMRESCycleResult gmres_ca_cycle(const SparseMatrixCSR& A,
                                  const Vector& b,
                                  const Vector& x_start,
                                  const Vector& r_start,
                                  Scalar beta,
                                  const GMRESConfig& config,
                                  const DenseBlock& recycled_U = DenseBlock(),
                                  const Vector& shifts = Vector());

}

#endif