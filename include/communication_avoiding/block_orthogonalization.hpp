/**
 * @file include/communication_avoiding/block_orthogonalization.hpp
 * @brief Declares sequential block orthogonalisation.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_HPP
#define COMMUNICATION_AVOIDING_BLOCK_ORTHOGONALIZATION_HPP

#include "common/dense_block.hpp"
#include "common/types.hpp"

namespace gmres {

/**
 * @brief Stores a sequential block orthogonalisation result.
 */
struct BlockOrthogonalizationResult {
    DenseBlock Q_block;
    DenseMatrix R_old;
    DenseMatrix R_block;
    Index accepted_columns = 0;
    bool truncated = false;
};

/**
 * @brief Orthogonalises a block with modified Gram-Schmidt.
 * @param old_basis Existing basis.
 * @param old_cols Number of leading basis columns to use.
 * @param input_block Block to orthogonalise.
 * @return Orthonormal block, recurrence factors, and acceptance status.
 */
BlockOrthogonalizationResult block_modified_gram_schmidt(const DenseBlock& old_basis,
                                                         Index old_cols,
                                                         const DenseBlock& input_block);

}

#endif
