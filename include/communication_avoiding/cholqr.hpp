/**
 * @file include/communication_avoiding/cholqr.hpp
 * @brief Declares sequential Cholesky QR.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_CHOLQR_HPP
#define COMMUNICATION_AVOIDING_CHOLQR_HPP

#include "common/dense_block.hpp"
#include "communication_avoiding/partial_cholesky.hpp"

namespace gmres {

/**
 * @brief Stores a Cholesky QR factorisation result.
 */
struct CholQRResult {
    DenseBlock Q;
    DenseBlock R;
    Index accepted_columns = 0;
    bool truncated = false;
};

/**
 * @brief Orthonormalises a dense block using Cholesky QR.
 * @param input_block Block to factor.
 * @param partial_cholesky_options Acceptance rule for the triangular factor.
 * @return Orthonormal block, triangular factor, and acceptance status.
 */
CholQRResult cholqr(
    const DenseBlock& input_block,
    const PartialCholeskyOptions& partial_cholesky_options =
        PartialCholeskyOptions{});

}

#endif
