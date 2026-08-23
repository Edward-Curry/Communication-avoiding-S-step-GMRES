/**
 * @file include/communication_avoiding/bcgs2_cholqr.hpp
 * @brief Declares BCGS2 with CholQR.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_BCGS2_CHOLQR_HPP
#define COMMUNICATION_AVOIDING_BCGS2_CHOLQR_HPP

#include "common/dense_block.hpp"
#include "communication_avoiding/block_orthogonalization.hpp"
#include "communication_avoiding/partial_cholesky.hpp"

namespace gmres {

/**
 * @brief Orthogonalises a block with two BCGS passes and CholQR.
 * @param old_basis Existing basis.
 * @param old_cols Number of leading basis columns to use.
 * @param input_block Block to orthogonalise.
 * @param partial_cholesky_options Acceptance rule for the CholQR factorisation.
 * @return Orthonormal block and recurrence factors.
 */
BlockOrthogonalizationResult bcgs2_cholqr(
    const DenseBlock& old_basis,
    Index old_cols,
    const DenseBlock& input_block,
    const PartialCholeskyOptions& partial_cholesky_options =
        PartialCholeskyOptions{});

}

#endif
