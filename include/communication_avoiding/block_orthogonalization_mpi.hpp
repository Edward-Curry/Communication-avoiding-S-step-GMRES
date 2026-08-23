/**
 * @file include/communication_avoiding/block_orthogonalization_mpi.hpp
 * @brief Declares MPI block orthogonalisation.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_MPI_HPP
#define COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_MPI_HPP

#include "common/dense_block.hpp"
#include "common/types.hpp"
#include "parallel/distributed_dense_block.hpp"

namespace gmres {

/**
 * @brief Stores a distributed block orthogonalisation result.
 */
struct BlockOrthogonalizationMPIResult {
    /// @brief Local rows of the accepted orthonormal block.
    DenseBlock Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

/**
 * @brief Orthogonalises a distributed block with modified Gram-Schmidt.
 * @param old_basis Existing distributed basis.
 * @param old_cols Number of leading basis columns to use.
 * @param input_block Distributed block to orthogonalise.
 * @return Orthonormal local block, recurrence factors, and acceptance status.
 */
BlockOrthogonalizationMPIResult block_modified_gram_schmidt_mpi(
    const DistributedDenseBlock& old_basis,
    Index old_cols,
    const DistributedDenseBlock& input_block
);

}

#endif
