/**
 * @file include/communication_avoiding/bcgs.hpp
 * @brief Declares block classical Gram-Schmidt.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_BCGS_HPP
#define COMMUNICATION_AVOIDING_BCGS_HPP

#include "common/dense_block.hpp"

namespace gmres {

/**
 * @brief Stores the result of one block Gram-Schmidt pass.
 */
struct BCGSPassResult {
    DenseBlock block;
    DenseBlock coefficients;
};

/**
 * @brief Projects a block away from an existing basis.
 * @param old_basis Basis containing the projection vectors.
 * @param old_cols Number of leading basis columns to use.
 * @param input_block Block to orthogonalise.
 * @return Projected block and projection coefficients.
 */
BCGSPassResult bcgs_pass(const DenseBlock& old_basis,
                         Index old_cols,
                         const DenseBlock& input_block);

}

#endif
