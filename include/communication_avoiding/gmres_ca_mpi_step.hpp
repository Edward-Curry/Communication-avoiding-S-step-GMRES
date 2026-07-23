#ifndef COMMUNICATION_AVOIDING_GMRES_CA_MPI_STEP_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_MPI_STEP_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "communication_avoiding/ca_residual_history.hpp"
#include "parallel/distributed_dense_block.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

struct CAGMRESMPICycleResult {
    DistributedVector x;
    CAResidualHistory residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;

    // The recycle_count newly generated blocks with the largest relative
    // residual drop this cycle, ranked best-first and concatenated into one
    // block (0 columns when config.enable_recycling is false or no block made
    // progress). Callers adopt this wholesale as the next cycle's recycled
    // subspace.
    DistributedDenseBlock recycle_candidate_block;
};

// recycled_U (may have 0 columns) supplies a subspace from a previous cycle
// to augment the start of this cycle's search space; see GMRESConfig::
// enable_recycling for the construction and cost.
CAGMRESMPICycleResult gmres_ca_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                         const DistributedVector& b,
                                         const DistributedVector& x_start,
                                         const DistributedVector& r_start,
                                         Scalar beta,
                                         const GMRESConfig& config,
                                         const DistributedDenseBlock& recycled_U =
                                             DistributedDenseBlock());

}

#endif