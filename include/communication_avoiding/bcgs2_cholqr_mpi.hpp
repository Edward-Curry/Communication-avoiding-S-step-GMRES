#ifndef COMMUNICATION_AVOIDING_BCGS2_CHOLQR_MPI_HPP
#define COMMUNICATION_AVOIDING_BCGS2_CHOLQR_MPI_HPP

#include "communication_avoiding/block_orthogonalization_mpi.hpp"

namespace gmres {

BlockOrthogonalizationMPIResult bcgs2_cholqr_mpi(
    const DistributedVectorList& old_basis,
    const DistributedVectorList& input_block);

}

#endif
