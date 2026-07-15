#ifndef COMMUNICATION_AVOIDING_BCGS_MPI_HPP
#define COMMUNICATION_AVOIDING_BCGS_MPI_HPP

#include "common/dense_block.hpp"
#include "parallel/distributed_dense_block.hpp"

namespace gmres {

struct BCGSMPIPassResult {
    DistributedDenseBlock block;
    DenseBlock coefficients;
};

// Orthogonalises input_block against the leading old_cols columns of
// old_basis, without copying the old basis. One MPI_Allreduce.
BCGSMPIPassResult bcgs_pass_mpi(
    const DistributedDenseBlock& old_basis,
    Index old_cols,
    const DistributedDenseBlock& input_block);

}

#endif
