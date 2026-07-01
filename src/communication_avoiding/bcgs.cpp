#include "communication_avoiding/bcgs.hpp"

#include <stdexcept>

namespace gmres {

BCGSPassResult bcgs_pass(const DenseBlock& old_basis,
                         const DenseBlock& input_block)
{
    if (old_basis.rows() != input_block.rows()) {
        throw std::invalid_argument("BCGS basis and input block row counts differ.");
    }

    BCGSPassResult result;
    result.block = input_block;
    result.coefficients = transpose_multiply(old_basis, input_block);
    subtract_product(result.block, old_basis, result.coefficients);
    return result;
}

}
