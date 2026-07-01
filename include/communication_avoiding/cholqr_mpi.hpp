#ifndef COMMUNICATION_AVOIDING_CHOLQR_MPI_HPP
#define COMMUNICATION_AVOIDING_CHOLQR_MPI_HPP

#include "common/dense_block.hpp"
#include "parallel/distributed_dense_block.hpp"

namespace gmres {

struct CholQRMPIResult {
    DistributedDenseBlock Q;
    DenseBlock R;
    Index accepted_columns = 0;
    bool truncated = false;
};

CholQRMPIResult cholqr_mpi(const DistributedDenseBlock& input_block);

}

#endif
