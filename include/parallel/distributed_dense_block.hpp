#ifndef PARALLEL_DISTRIBUTED_DENSE_BLOCK_HPP
#define PARALLEL_DISTRIBUTED_DENSE_BLOCK_HPP

#include "common/dense_block.hpp"
#include "parallel/distributed_vector.hpp"

#include <mpi.h>

namespace gmres {

class DistributedDenseBlock {
public:
    DistributedDenseBlock() = default;

    DistributedDenseBlock(Index global_rows,
                          Index local_start,
                          DenseBlock local_block,
                          MPI_Comm comm);

    Index global_rows() const;
    Index local_start() const;
    Index local_rows() const;
    Index cols() const;
    MPI_Comm communicator() const;

    DenseBlock& local_block();
    const DenseBlock& local_block() const;

    DistributedDenseBlock leading_columns(Index count) const;

private:
    Index global_rows_ = 0;
    Index local_start_ = 0;
    DenseBlock local_block_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
};

DistributedDenseBlock pack_distributed_columns(
    const DistributedVectorList& columns);

DistributedVectorList unpack_distributed_columns(
    const DistributedDenseBlock& block);

void check_compatible(const DistributedDenseBlock& left,
                      const DistributedDenseBlock& right);

}

#endif
