#ifndef COMMUNICATION_AVOIDING_GMRES_CA_MPI_STEP_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_MPI_STEP_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "communication_avoiding/ca_residual_history.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

struct CAGMRESMPICycleResult {
    DistributedVector x;
    CAResidualHistory residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;
};

CAGMRESMPICycleResult gmres_ca_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                         const DistributedVector& b,
                                         const DistributedVector& x_start,
                                         const DistributedVector& r_start,
                                         Scalar beta,
                                         const GMRESConfig& config);

}

#endif