#include "communication_avoiding/bcgs.hpp"

#include <stdexcept>

namespace gmres {

BCGSPassResult bcgs_pass(const DenseBlock& old_basis,
                         Index old_cols,
                         const DenseBlock& input_block)
{
    if (old_basis.rows() != input_block.rows()) {
        throw std::invalid_argument("BCGS basis and input block row counts differ.");
    }

    if (old_cols > old_basis.cols()) {
        throw std::invalid_argument("BCGS old_cols exceeds the basis width.");
    }

    BCGSPassResult result;
    result.block = input_block;
    result.coefficients = transpose_multiply(old_basis, old_cols, input_block);
    subtract_product(result.block, old_basis, old_cols, result.coefficients);
    return result;
}

}
