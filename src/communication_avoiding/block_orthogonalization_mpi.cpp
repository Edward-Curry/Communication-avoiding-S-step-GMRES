#include "communication_avoiding/block_orthogonalization_mpi.hpp"

#include <cblas.h>
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace gmres {

namespace {

Scalar allreduce_scalar(Scalar local, MPI_Comm comm)
{
    Scalar global = 0.0;
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, comm);
    return global;
}

void truncate_r_factors(DenseMatrix& R_old, DenseMatrix& R_block, Index accepted)
{
    for (Vector& row : R_old) {
        row.resize(accepted);
    }

    R_block.resize(accepted);

    for (Vector& row : R_block) {
        row.resize(accepted);
    }
}

}

BlockOrthogonalizationMPIResult block_modified_gram_schmidt_mpi(
    const DistributedDenseBlock& old_basis,
    Index old_cols,
    const DistributedDenseBlock& input_block
)
{
    if (input_block.cols() == 0) {
        throw std::invalid_argument("block_modified_gram_schmidt_mpi: input_block is empty.");
    }

    check_compatible(old_basis, input_block);

    if (old_cols > old_basis.cols()) {
        throw std::invalid_argument("block_modified_gram_schmidt_mpi: old_cols exceeds the basis width.");
    }

    const DenseBlock& old_local = old_basis.local_block();
    const DenseBlock& input_local = input_block.local_block();
    MPI_Comm comm = input_block.communicator();

    const Index rows = input_local.rows();
    const Index block_size = input_local.cols();
    const int n = static_cast<int>(rows);

    BlockOrthogonalizationMPIResult result;
    result.R_old.assign(old_cols, Vector(block_size, 0.0));
    result.R_block.assign(block_size, Vector(block_size, 0.0));

    DenseBlock Q(rows, block_size);
    Index accepted = 0;

    Vector w(rows);

    for (Index j = 0; j < block_size; ++j) {
        std::copy(input_local.column(j), input_local.column(j) + rows, w.begin());

        // First: orthogonalise this column against the old distributed basis.
        for (Index i = 0; i < old_cols; ++i) {
            const Scalar local_dot = cblas_ddot(n, old_local.column(i), 1, w.data(), 1);
            const Scalar coefficient = allreduce_scalar(local_dot, comm);
            result.R_old[i][j] = coefficient;
            cblas_daxpy(n, -coefficient, old_local.column(i), 1, w.data(), 1);
        }

        // Second: orthogonalise against previous columns in this new block.
        for (Index i = 0; i < accepted; ++i) {
            const Scalar local_dot = cblas_ddot(n, Q.column(i), 1, w.data(), 1);
            const Scalar coefficient = allreduce_scalar(local_dot, comm);
            result.R_block[i][j] = coefficient;
            cblas_daxpy(n, -coefficient, Q.column(i), 1, w.data(), 1);
        }

        // Third: normalise this new column.
        const Scalar local_sumsq = cblas_ddot(n, w.data(), 1, w.data(), 1);
        const Scalar remaining_norm = std::sqrt(allreduce_scalar(local_sumsq, comm));

        if (remaining_norm == 0.0) {
            truncate_r_factors(result.R_old, result.R_block, accepted);
            result.Q_block = Q.leading_columns(accepted);
            result.accepted_columns = accepted;
            result.truncated = true;
            return result;
        }

        result.R_block[j][j] = remaining_norm;
        cblas_dscal(n, 1.0 / remaining_norm, w.data(), 1);
        Q.set_column(j, w);
        ++accepted;
    }

    result.Q_block = std::move(Q);
    result.accepted_columns = accepted;
    result.truncated = false;

    return result;
}

}
