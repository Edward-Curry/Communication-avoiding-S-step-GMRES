/**
 * @file tests/test_sstep_arnoldi_mpi.cpp
 * @brief Tests distributed s-step Arnoldi basis construction.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "communication_avoiding/sstep_arnoldi_mpi.hpp"

#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <cassert>
#include <cmath>
#include <mpi.h>
#include <print>
#include <vector>

namespace {

/**
 * @brief Returns the contiguous local size assigned to an MPI rank.
 * @param global_size Global vector length.
 * @param rank MPI rank.
 * @param size Number of MPI ranks.
 * @return Number of locally owned entries.
 */
gmres::Index local_size_for_rank(gmres::Index global_size, int rank, int size)
{
    const gmres::Index base = global_size / static_cast<gmres::Index>(size);
    const gmres::Index remainder = global_size % static_cast<gmres::Index>(size);

    if (static_cast<gmres::Index>(rank) < remainder) {
        return base + 1;
    }

    return base;
}

/**
 * @brief Returns the first global index assigned to an MPI rank.
 * @param global_size Global vector length.
 * @param rank MPI rank.
 * @param size Number of MPI ranks.
 * @return First locally owned global index.
 */
gmres::Index local_start_for_rank(gmres::Index global_size, int rank, int size)
{
    gmres::Index start = 0;

    for (int r = 0; r < rank; ++r) {
        start += local_size_for_rank(global_size, r, size);
    }

    return start;
}

/**
 * @brief Tests two scalar values for approximate equality.
 * @param a First value.
 * @param b Second value.
 * @param tolerance Permitted absolute difference.
 * @return True when the values are within the tolerance.
 */
bool nearly_equal(gmres::Scalar a, gmres::Scalar b, gmres::Scalar tolerance)
{
    return std::abs(a - b) < tolerance;
}

}

/**
 * @brief Runs the distributed s-step Arnoldi test.
 * @param argc Number of command-line arguments passed to MPI.
 * @param argv Command-line arguments passed to MPI.
 * @return Zero when all ranks pass their assertions.
 */
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    using namespace gmres;

    const Index global_size = 4;
    const Index local_rows = local_size_for_rank(global_size, rank, size);
    const Index local_start = local_start_for_rank(global_size, rank, size);

    // A diagonal matrix gives a predictable distributed SpMV result.
    Vector values;
    std::vector<Index> col_indices;
    std::vector<Index> row_ptr;

    row_ptr.push_back(0);

    for (Index local_row = 0; local_row < local_rows; ++local_row) {
        const Index global_row = local_start + local_row;

        values.push_back(static_cast<Scalar>(global_row + 1));
        col_indices.push_back(global_row);
        row_ptr.push_back(values.size());
    }

    DistributedSparseMatrixCSR A(
        global_size,
        global_size,
        local_start,
        local_rows,
        values,
        col_indices,
        row_ptr,
        MPI_COMM_WORLD
    );

    // The starting vector has unit Euclidean norm.
    Vector local_q_values(local_rows, 0.5);

    DistributedVector q(
        global_size,
        local_start,
        local_q_values,
        MPI_COMM_WORLD
    );

    DenseBlock local_basis(local_rows, 1);
    local_basis.set_column(0, local_q_values);

    const DistributedDenseBlock old_basis(global_size,
                                          local_start,
                                          local_basis,
                                          MPI_COMM_WORLD);

    const Index s = 2;

    const std::vector<BlockOrthogonalizationMethod> methods{
        BlockOrthogonalizationMethod::ModifiedGramSchmidt,
        BlockOrthogonalizationMethod::BCGS2CholQR
    };

    for (BlockOrthogonalizationMethod method : methods) {
        SStepArnoldiMPIResult result =
            sstep_arnoldi_block_mpi(A,
                                    old_basis,
                                    1,
                                    s,
                                    PolynomialBasisType::Monomial,
                                    method);

        assert(result.accepted_columns > 0);
        assert(result.Q_block.cols() == result.accepted_columns);

        auto column_vector = [&](Index col) {
            return DistributedVector(global_size,
                                     local_start,
                                     result.Q_block.get_column(col),
                                     MPI_COMM_WORLD);
        };

        for (Index col = 0; col < result.Q_block.cols(); ++col) {
            const DistributedVector v = column_vector(col);
            assert(nearly_equal(norm2_mpi(v), 1.0, 1e-10));
            assert(nearly_equal(dot_mpi(q, v), 0.0, 1e-10));
        }

        for (Index i = 0; i < result.Q_block.cols(); ++i) {
            for (Index j = i + 1; j < result.Q_block.cols(); ++j) {
                assert(nearly_equal(dot_mpi(column_vector(i), column_vector(j)), 0.0, 1e-10));
            }
        }
    }

    if (rank == 0) {
        std::println("test_sstep_arnoldi_mpi passed for both block orthogonalization methods.");
    }

    MPI_Finalize();

    return 0;
}
