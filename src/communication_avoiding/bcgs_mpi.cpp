/**
 * @file src/communication_avoiding/bcgs_mpi.cpp
 * @brief Implements one distributed BCGS projection pass.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "communication_avoiding/bcgs_mpi.hpp"

#include <limits>
#include <mpi.h>
#include <stdexcept>

namespace gmres {

namespace {

int mpi_count(Index count)
{
    if (count > static_cast<Index>(std::numeric_limits<int>::max())) {
        throw std::length_error("BCGS coefficient matrix exceeds the MPI count range.");
    }

    return static_cast<int>(count);
}

}

BCGSMPIPassResult bcgs_pass_mpi(
    const DistributedDenseBlock& old_basis,
    Index old_cols,
    const DistributedDenseBlock& input_block)
{
    check_compatible(old_basis, input_block);

    if (old_cols > old_basis.cols()) {
        throw std::invalid_argument("BCGS old_cols exceeds the basis width.");
    }

    BCGSMPIPassResult result;
    result.block = input_block;
    result.coefficients = transpose_multiply(old_basis.local_block(),
                                             old_cols,
                                             input_block.local_block());

    if (result.coefficients.size() > 0) {
        MPI_Allreduce(MPI_IN_PLACE,
                      result.coefficients.data(),
                      mpi_count(result.coefficients.size()),
                      MPI_DOUBLE,
                      MPI_SUM,
                      input_block.communicator());
    }

    subtract_product(result.block.local_block(),
                     old_basis.local_block(),
                     old_cols,
                     result.coefficients);

    return result;
}

}
