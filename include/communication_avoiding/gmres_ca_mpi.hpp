#ifndef COMMUNICATION_AVOIDING_GMRES_CA_MPI_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_MPI_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

struct CAGMRESMPIResult {
    DistributedVector x;
    Vector residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;
};

CAGMRESMPIResult gmres_ca_mpi(const DistributedSparseMatrixCSR& A,
                              const DistributedVector& b,
                              const DistributedVector& x0,
                              const GMRESConfig& config);

}

#endif