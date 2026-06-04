#ifndef PARALLEL_GMRES_MPI_HPP
#define PARALLEL_GMRES_MPI_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    struct GMRESMPIResult
    {
        DistributedVector x;
        Vector residual_history;
        Index iterations = 0;
        bool converged = false;
    };

    GMRESMPIResult gmres_mpi(const DistributedSparseMatrixCSR& A,
                             const DistributedVector& b,
                             const DistributedVector& x0,
                             const GMRESConfig& config);
}

#endif