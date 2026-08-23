/**
 * @file include/communication_avoiding/bcgs2_cholqr_mpi.hpp
 * @brief Declares MPI BCGS2 with CholQR.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_BCGS2_CHOLQR_MPI_HPP
#define COMMUNICATION_AVOIDING_BCGS2_CHOLQR_MPI_HPP

#include "communication_avoiding/block_orthogonalization_mpi.hpp"
#include "communication_avoiding/partial_cholesky.hpp"
#include "parallel/distributed_dense_block.hpp"

namespace gmres {

/**
 * @brief Orthogonalises a distributed block with MPI BCGS2 and CholQR.
 * @param old_basis Existing distributed basis.
 * @param old_cols Number of leading basis columns to use.
 * @param input_block Distributed block to orthogonalise.
 * @param partial_cholesky_options Acceptance rule for the CholQR factorisation.
 * @return Orthonormal distributed block and replicated recurrence factors.
 */
BlockOrthogonalizationMPIResult bcgs2_cholqr_mpi(
    const DistributedDenseBlock& old_basis,
    Index old_cols,
    const DistributedDenseBlock& input_block,
    const PartialCholeskyOptions& partial_cholesky_options =
        PartialCholeskyOptions{});

}

#endif
