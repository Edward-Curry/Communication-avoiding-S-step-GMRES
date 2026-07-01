#include "communication_avoiding/cholqr.hpp"

#include "communication_avoiding/partial_cholesky.hpp"

#include <stdexcept>

namespace gmres {

CholQRResult cholqr(const DenseBlock& input_block)
{
    if (input_block.cols() == 0) {
        throw std::invalid_argument("CholQR input block is empty.");
    }

    PartialCholeskyResult factor =
        partial_cholesky(gram_matrix(input_block));

    CholQRResult result;
    result.R = factor.R;
    result.accepted_columns = factor.accepted_columns;
    result.truncated = factor.truncated;

    if (factor.accepted_columns == 0) {
        result.Q = DenseBlock(input_block.rows(), 0);
        return result;
    }

    result.Q = input_block.leading_columns(factor.accepted_columns);
    right_solve_upper(result.Q, result.R);
    return result;
}

}
