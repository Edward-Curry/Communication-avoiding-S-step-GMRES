#include "communication_avoiding/block_orthogonalization.hpp"

#include "common/vector_ops.hpp"

#include <stdexcept>

namespace gmres {

BlockOrthogonalizationResult block_modified_gram_schmidt(const VectorList& old_basis,
                                                         const VectorList& input_block)
{
    if (input_block.empty()) {
        throw std::invalid_argument("block_modified_gram_schmidt: input_block is empty.");
    }

    const Index block_size = input_block.size();
    const Index old_size = old_basis.size();
    const Index vector_size = input_block[0].size();

    for (const Vector& v : old_basis) {
        if (v.size() != vector_size) {
            throw std::invalid_argument("block_modified_gram_schmidt: old_basis vector has wrong size.");
        }
    }

    for (const Vector& w : input_block) {
        if (w.size() != vector_size) {
            throw std::invalid_argument("block_modified_gram_schmidt: input_block vector has wrong size.");
        }
    }

    BlockOrthogonalizationResult result;

    result.R_old.assign(old_size, Vector(block_size, 0.0));
    result.R_block.assign(block_size, Vector(block_size, 0.0));
    result.Q_block.reserve(block_size);

    for (Index j = 0; j < block_size; ++j) {
        Vector w = input_block[j];

        // First: orthogonalise this vector against the old basis.
        for (Index i = 0; i < old_size; ++i) {
            const Scalar coefficient = dot(old_basis[i], w);
            result.R_old[i][j] = coefficient;
            axpy(-coefficient, old_basis[i], w);
        }

        // Second: orthogonalise this vector against previous vectors in this new block.
        for (Index i = 0; i < result.Q_block.size(); ++i) {
            const Scalar coefficient = dot(result.Q_block[i], w);
            result.R_block[i][j] = coefficient;
            axpy(-coefficient, result.Q_block[i], w);
        }

        // Third: normalize this new vector.
        const Scalar remaining_norm = norm2(w);

        if (remaining_norm == 0.0) {
            result.truncated = true;
            result.accepted_columns = result.Q_block.size();
            return result;
        }

        result.R_block[j][j] = remaining_norm;
        scal(1.0 / remaining_norm, w);
        result.Q_block.push_back(w);
    }

    result.accepted_columns = result.Q_block.size();
    result.truncated = false;

    return result;
}

}