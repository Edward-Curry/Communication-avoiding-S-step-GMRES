/**
 * @file include/parallel/distributed_dense_block.hpp
 * @brief Declares row-distributed dense blocks.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_DISTRIBUTED_DENSE_BLOCK_HPP
#define PARALLEL_DISTRIBUTED_DENSE_BLOCK_HPP

#include "common/dense_block.hpp"
#include "parallel/distributed_vector.hpp"

#include <mpi.h>

namespace gmres {

/**
 * @brief Stores a dense block distributed by contiguous row ranges.
 */
class DistributedDenseBlock {
public:
    /** @brief Constructs an empty distributed block. */
    DistributedDenseBlock() = default;

    /**
     * @brief Constructs a distributed block from local storage.
     * @param global_rows Global row count.
     * @param local_start Global index of the first local row.
     * @param local_block Local rows in column-major storage.
     * @param comm Communicator defining the distribution.
     */
    DistributedDenseBlock(Index global_rows,
                          Index local_start,
                          DenseBlock local_block,
                          MPI_Comm comm);

    /** @return Global row count. */
    Index global_rows() const;
    /** @return Global index of the first local row. */
    Index local_start() const;
    /** @return Number of local rows. */
    Index local_rows() const;
    /** @return Common column count. */
    Index cols() const;
    /** @return Communicator defining the row distribution. */
    MPI_Comm communicator() const;

    /** @return Mutable local dense block. */
    DenseBlock& local_block();
    /** @return Const local dense block. */
    const DenseBlock& local_block() const;

    /**
     * @brief Copies leading distributed columns.
     * @param count Number of columns to copy.
     * @return Block containing the leading columns.
     */
    DistributedDenseBlock leading_columns(Index count) const;

private:
    Index global_rows_ = 0;
    Index local_start_ = 0;
    DenseBlock local_block_;
    MPI_Comm comm_ = MPI_COMM_WORLD;
};

/**
 * @brief Packs equally distributed vectors into a dense block.
 * @param columns Distributed vectors with matching layouts.
 * @return Row-distributed dense block.
 */
DistributedDenseBlock pack_distributed_columns(
    const DistributedVectorList& columns);

/**
 * @brief Copies a distributed block into individual vectors.
 * @param block Source block.
 * @return Distributed vector list containing block columns.
 */
DistributedVectorList unpack_distributed_columns(
    const DistributedDenseBlock& block);

/**
 * @brief Validates matching distributed-block layouts.
 * @param left First block.
 * @param right Second block.
 */
void check_compatible(const DistributedDenseBlock& left,
                      const DistributedDenseBlock& right);

}

#endif
