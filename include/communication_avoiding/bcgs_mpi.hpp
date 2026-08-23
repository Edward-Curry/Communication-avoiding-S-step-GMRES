/**
 * @file include/communication_avoiding/bcgs_mpi.hpp
 * @brief Declares MPI block classical Gram-Schmidt.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_BCGS_MPI_HPP
#define COMMUNICATION_AVOIDING_BCGS_MPI_HPP

#include "common/dense_block.hpp"
#include "parallel/distributed_dense_block.hpp"

namespace gmres {

/**
 * @brief Stores the result of one distributed BCGS pass.
 */
struct BCGSMPIPassResult {
    DistributedDenseBlock block;
    DenseBlock coefficients;
};

/**
 * @brief Projects a distributed block away from an existing basis.
 * @param old_basis Distributed basis containing projection vectors.
 * @param old_cols Number of leading basis columns to use.
 * @param input_block Distributed block to orthogonalise.
 * @return Projected block and replicated projection coefficients.
 */
BCGSMPIPassResult bcgs_pass_mpi(
    const DistributedDenseBlock& old_basis,
    Index old_cols,
    const DistributedDenseBlock& input_block);

}

#endif
