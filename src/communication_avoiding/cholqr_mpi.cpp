#include "communication_avoiding/cholqr_mpi.hpp"

#include "communication_avoiding/partial_cholesky.hpp"

#include <limits>
#include <mpi.h>
#include <stdexcept>

namespace gmres {

namespace {

int mpi_count(Index count)
{
    if (count > static_cast<Index>(std::numeric_limits<int>::max())) {
        throw std::length_error("CholQR Gram matrix exceeds the MPI count range.");
    }

    return static_cast<int>(count);
}

}

CholQRMPIResult cholqr_mpi(
    const DistributedDenseBlock& input_block,
    const PartialCholeskyOptions& partial_cholesky_options)
{
    if (input_block.cols() == 0) {
        throw std::invalid_argument("MPI CholQR input block is empty.");
    }

    DenseBlock gram = gram_matrix(input_block.local_block());

    MPI_Allreduce(MPI_IN_PLACE,
                  gram.data(),
                  mpi_count(gram.size()),
                  MPI_DOUBLE,
                  MPI_SUM,
                  input_block.communicator());

    PartialCholeskyResult factor =
        partial_cholesky(gram, partial_cholesky_options);

    CholQRMPIResult result;
    result.R = factor.R;
    result.accepted_columns = factor.accepted_columns;
    result.truncated = factor.truncated;

    DenseBlock local_q =
        input_block.local_block().leading_columns(factor.accepted_columns);

    if (factor.accepted_columns > 0) {
        right_solve_upper(local_q, result.R);
    }

    result.Q = DistributedDenseBlock(input_block.global_rows(),
                                     input_block.local_start(),
                                     std::move(local_q),
                                     input_block.communicator());

    return result;
}

}
