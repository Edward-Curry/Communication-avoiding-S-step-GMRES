#ifndef COMMUNICATION_AVOIDING_BCGS2_CHOLQR_MPI_HPP
#define COMMUNICATION_AVOIDING_BCGS2_CHOLQR_MPI_HPP

#include "communication_avoiding/block_orthogonalization_mpi.hpp"
#include "parallel/distributed_dense_block.hpp"

namespace gmres {

// Two-pass block classical Gram-Schmidt with CholQR intra-block
// orthonormalisation, against the leading old_cols columns of old_basis.
// Four MPI_Allreduce calls per block, independent of the block width.
BlockOrthogonalizationMPIResult bcgs2_cholqr_mpi(
    const DistributedDenseBlock& old_basis,
    Index old_cols,
    const DistributedDenseBlock& input_block);

}

#endif
