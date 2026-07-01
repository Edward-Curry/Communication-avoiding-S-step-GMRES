#include "parallel/distributed_dense_block.hpp"

#include <stdexcept>
#include <utility>

namespace gmres {

DistributedDenseBlock::DistributedDenseBlock(Index global_rows,
                                             Index local_start,
                                             DenseBlock local_block,
                                             MPI_Comm comm)
    : global_rows_(global_rows),
      local_start_(local_start),
      local_block_(std::move(local_block)),
      comm_(comm)
{
    if (local_start_ > global_rows_
        || local_start_ + local_block_.rows() > global_rows_) {
        throw std::invalid_argument("Distributed dense block local range is invalid.");
    }
}

Index DistributedDenseBlock::global_rows() const
{
    return global_rows_;
}

Index DistributedDenseBlock::local_start() const
{
    return local_start_;
}

Index DistributedDenseBlock::local_rows() const
{
    return local_block_.rows();
}

Index DistributedDenseBlock::cols() const
{
    return local_block_.cols();
}

MPI_Comm DistributedDenseBlock::communicator() const
{
    return comm_;
}

DenseBlock& DistributedDenseBlock::local_block()
{
    return local_block_;
}

const DenseBlock& DistributedDenseBlock::local_block() const
{
    return local_block_;
}

DistributedDenseBlock DistributedDenseBlock::leading_columns(Index count) const
{
    return DistributedDenseBlock(global_rows_,
                                 local_start_,
                                 local_block_.leading_columns(count),
                                 comm_);
}

DistributedDenseBlock pack_distributed_columns(
    const DistributedVectorList& columns)
{
    if (columns.empty()) {
        throw std::invalid_argument("Cannot pack an empty distributed vector list.");
    }

    const DistributedVector& first = columns.front();
    VectorList local_columns;
    local_columns.reserve(columns.size());

    for (const DistributedVector& column : columns) {
        if (column.global_size() != first.global_size()
            || column.local_start() != first.local_start()
            || column.local_size() != first.local_size()
            || column.communicator() != first.communicator()) {
            throw std::invalid_argument("Distributed vector columns are incompatible.");
        }

        local_columns.push_back(column.local_values());
    }

    return DistributedDenseBlock(first.global_size(),
                                 first.local_start(),
                                 pack_columns(local_columns),
                                 first.communicator());
}

DistributedVectorList unpack_distributed_columns(
    const DistributedDenseBlock& block)
{
    VectorList local_columns = unpack_columns(block.local_block());
    DistributedVectorList result;
    result.reserve(local_columns.size());

    for (Vector& column : local_columns) {
        result.emplace_back(block.global_rows(),
                            block.local_start(),
                            std::move(column),
                            block.communicator());
    }

    return result;
}

void check_compatible(const DistributedDenseBlock& left,
                      const DistributedDenseBlock& right)
{
    if (left.global_rows() != right.global_rows()
        || left.local_start() != right.local_start()
        || left.local_rows() != right.local_rows()
        || left.communicator() != right.communicator()) {
        throw std::invalid_argument("Distributed dense blocks are incompatible.");
    }
}

}
