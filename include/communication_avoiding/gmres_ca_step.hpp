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
};

// recycled_U (may have 0 columns) supplies a subspace from a previous cycle
// to augment the start of this cycle's search space; see GMRESConfig::
// enable_recycling for the construction and cost.
CAGMRESCycleResult gmres_ca_cycle(const SparseMatrixCSR& A,
                                  const Vector& b,
                                  const Vector& x_start,
                                  const Vector& r_start,
                                  Scalar beta,
                                  const GMRESConfig& config,
                                  const DenseBlock& recycled_U = DenseBlock());

}

#endif