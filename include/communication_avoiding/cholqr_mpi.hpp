/**
 * @file include/communication_avoiding/cholqr_mpi.hpp
 * @brief Declares MPI Cholesky QR.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_CHOLQR_MPI_HPP
#define COMMUNICATION_AVOIDING_CHOLQR_MPI_HPP

#include "common/dense_block.hpp"
#include "communication_avoiding/partial_cholesky.hpp"
#include "parallel/distributed_dense_block.hpp"

namespace gmres {

/**
 * @brief Stores a distributed Cholesky QR factorisation result.
 */
struct CholQRMPIResult {
    DistributedDenseBlock Q;
    DenseBlock R;
    Index accepted_columns = 0;
    bool truncated = false;
};

/**
 * @brief Orthonormalises a distributed block using Cholesky QR.
 * @param input_block Distributed block to factor.
 * @param partial_cholesky_options Acceptance rule for the triangular factor.
 * @return Distributed orthonormal block, replicated factor, and status.
 */
CholQRMPIResult cholqr_mpi(
    const DistributedDenseBlock& input_block,
    const PartialCholeskyOptions& partial_cholesky_options =
        PartialCholeskyOptions{});

}

#endif
