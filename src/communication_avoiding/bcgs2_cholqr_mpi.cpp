#include "communication_avoiding/bcgs2_cholqr_mpi.hpp"

#include "common/dense_block.hpp"
#include "communication_avoiding/bcgs_mpi.hpp"
#include "communication_avoiding/cholqr_mpi.hpp"

#include <stdexcept>
#include <utility>

namespace gmres {

BlockOrthogonalizationMPIResult bcgs2_cholqr_mpi(
    const DistributedDenseBlock& old_basis,
    Index old_cols,
    const DistributedDenseBlock& input_block)
{
    if (old_cols == 0) {
        throw std::invalid_argument("bcgs2_cholqr_mpi: old basis is empty.");
    }

    if (input_block.cols() == 0) {
        throw std::invalid_argument("bcgs2_cholqr_mpi: input block is empty.");
    }

    check_compatible(old_basis, input_block);

    BCGSMPIPassResult first_bcgs = bcgs_pass_mpi(old_basis, old_cols, input_block);
    CholQRMPIResult first_cholqr = cholqr_mpi(first_bcgs.block);

    BlockOrthogonalizationMPIResult result;

    if (first_cholqr.accepted_columns == 0) {
        result.truncated = true;
        return result;
    }

    BCGSMPIPassResult second_bcgs =
        bcgs_pass_mpi(old_basis, old_cols, first_cholqr.Q);
    CholQRMPIResult second_cholqr = cholqr_mpi(second_bcgs.block);

    const Index accepted = second_cholqr.accepted_columns;
    result.accepted_columns = accepted;
    result.truncated = first_cholqr.truncated
        || second_cholqr.truncated
        || accepted < input_block.cols();

    if (accepted == 0) {
        return result;
    }

    DenseBlock first_coefficients =
        first_bcgs.coefficients.leading_columns(accepted);
    DenseBlock second_coefficients =
        second_bcgs.coefficients.leading_columns(accepted);
    DenseBlock first_factor =
        first_cholqr.R.leading_principal_block(accepted);

    DenseBlock correction = multiply(second_coefficients, first_factor);
    add_in_place(first_coefficients, correction);

    DenseBlock block_factor = multiply(second_cholqr.R, first_factor);

    result.Q_block = std::move(second_cholqr.Q.local_block());
    result.R_old = to_dense_matrix(first_coefficients);
    result.R_block = to_dense_matrix(block_factor);
    return result;
}

}
