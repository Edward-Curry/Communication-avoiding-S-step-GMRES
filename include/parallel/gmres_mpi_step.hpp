#ifndef PARALLEL_GMRES_MPI_STEP_HPP
#define PARALLEL_GMRES_MPI_STEP_HPP

#include "common/config.hpp"
#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    struct GMRESMPICycleResult
    {
        DistributedVector x;
        Vector residual_history;
        Index iterations = 0;
        bool converged = false;
    };

    GMRESMPICycleResult gmres_mpi_cycle(const DistributedSparseMatrixCSR& A,
                                        const DistributedVector& b,
                                        const DistributedVector& x_start,
                                        const DistributedVector& r_start,
                                        Scalar beta,
                                        const GMRESConfig& config);
}

#endif

