#include "communication_avoiding/block_orthogonalization_mpi.hpp"

#include "parallel/distributed_vector_ops.hpp"

#include <stdexcept>

namespace gmres {

BlockOrthogonalizationMPIResult block_modified_gram_schmidt_mpi(
    const DistributedVectorList& old_basis,
    const DistributedVectorList& input_block
)
{
    if (input_block.empty()) {
        throw std::invalid_argument("block_modified_gram_schmidt_mpi: input_block is empty.");
    }

    const Index block_size = input_block.size();
    const Index old_size = old_basis.size();

    for (const DistributedVector& v : old_basis) {
        check_compatible(v, input_block[0]);
    }

    for (const DistributedVector& w : input_block) {
        check_compatible(w, input_block[0]);
    }

    BlockOrthogonalizationMPIResult result;

    result.R_old.assign(old_size, Vector(block_size, 0.0));
    result.R_block.assign(block_size, Vector(block_size, 0.0));
    result.Q_block.reserve(block_size);

    for (Index j = 0; j < block_size; ++j) {
        DistributedVector w = input_block[j];

        // First: orthogonalise this vector against the old distributed basis.
        for (Index i = 0; i < old_size; ++i) {
            const Scalar coefficient = dot_mpi(old_basis[i], w);
            result.R_old[i][j] = coefficient;
            axpy_local(-coefficient, old_basis[i], w);
        }

        // Second: orthogonalise this vector against previous vectors in this new block.
        for (Index i = 0; i < result.Q_block.size(); ++i) {
            const Scalar coefficient = dot_mpi(result.Q_block[i], w);
            result.R_block[i][j] = coefficient;
            axpy_local(-coefficient, result.Q_block[i], w);
        }

        // Third: normalize this new distributed vector.
        const Scalar remaining_norm = norm2_mpi(w);

        if (remaining_norm == 0.0) {
            result.truncated = true;
            result.accepted_columns = result.Q_block.size();
            return result;
        }

        result.R_block[j][j] = remaining_norm;
        scal_local(1.0 / remaining_norm, w);
        result.Q_block.push_back(w);
    }

    result.accepted_columns = result.Q_block.size();
    result.truncated = false;

    return result;
}

}