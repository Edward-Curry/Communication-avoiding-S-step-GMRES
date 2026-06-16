#ifndef COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_MPI_HPP
#define COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_MPI_HPP

#include "common/types.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres {

struct BlockOrthogonalizationMPIResult {
    DistributedVectorList Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

BlockOrthogonalizationMPIResult block_modified_gram_schmidt_mpi(
    const DistributedVectorList& old_basis,
    const DistributedVectorList& input_block
);

}

#endif