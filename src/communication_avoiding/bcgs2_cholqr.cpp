#include "communication_avoiding/bcgs2_cholqr.hpp"

#include "common/dense_block.hpp"
#include "communication_avoiding/bcgs.hpp"
#include "communication_avoiding/cholqr.hpp"

#include <stdexcept>

namespace gmres {

BlockOrthogonalizationResult bcgs2_cholqr(
    const VectorList& old_basis,
    const VectorList& input_block)
{
    if (old_basis.empty()) {
        throw std::invalid_argument("bcgs2_cholqr: old basis is empty.");
    }

    if (input_block.empty()) {
        throw std::invalid_argument("bcgs2_cholqr: input block is empty.");
    }

    DenseBlock old_block = pack_columns(old_basis);
    DenseBlock input = pack_columns(input_block);

    if (old_block.rows() != input.rows()) {
        throw std::invalid_argument("bcgs2_cholqr: basis and input row counts differ.");
    }

    BCGSPassResult first_bcgs = bcgs_pass(old_block, input);
    CholQRResult first_cholqr = cholqr(first_bcgs.block);

    BlockOrthogonalizationResult result;

    if (first_cholqr.accepted_columns == 0) {
        result.truncated = true;
        return result;
    }

    BCGSPassResult second_bcgs = bcgs_pass(old_block, first_cholqr.Q);
    CholQRResult second_cholqr = cholqr(second_bcgs.block);

    const Index accepted = second_cholqr.accepted_columns;
    result.accepted_columns = accepted;
    result.truncated = first_cholqr.truncated
        || second_cholqr.truncated
        || accepted < input.cols();

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

    result.Q_block = unpack_columns(second_cholqr.Q);
    result.R_old = to_dense_matrix(first_coefficients);
    result.R_block = to_dense_matrix(block_factor);
    return result;
}

}
