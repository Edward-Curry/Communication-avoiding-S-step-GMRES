#include "communication_avoiding/gmres_ca_mpi.hpp"

#include "common/config.hpp"
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

gmres::Index local_size_for_rank(gmres::Index global_size, int rank, int size)
{
    const gmres::Index base = global_size / static_cast<gmres::Index>(size);
    const gmres::Index remainder = global_size % static_cast<gmres::Index>(size);

    if (static_cast<gmres::Index>(rank) < remainder) {
        return base + 1;
    }

    return base;
}

gmres::Index local_start_for_rank(gmres::Index global_size, int rank, int size)
{
    gmres::Index start = 0;

    for (int r = 0; r < rank; ++r) {
        start += local_size_for_rank(global_size, r, size);
    }

    return start;
}

bool nearly_equal(gmres::Scalar a, gmres::Scalar b, gmres::Scalar tolerance)
{
    return std::abs(a - b) < tolerance;
}

}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    using namespace gmres;

    /*
        Solve:

        [4 1] [x0] = [6]
        [1 3] [x1]   [7]

        Exact solution:
        x = [1, 2]
    */

    const Index global_size = 2;
    const Index local_rows = local_size_for_rank(global_size, rank, size);
    const Index local_start = local_start_for_rank(global_size, rank, size);

    Vector values;
    std::vector<Index> col_indices;
    std::vector<Index> row_ptr;

    row_ptr.push_back(0);

    for (Index local_row = 0; local_row < local_rows; ++local_row) {
        const Index global_row = local_start + local_row;

        if (global_row == 0) {
            values.push_back(4.0);
            col_indices.push_back(0);

            values.push_back(1.0);
            col_indices.push_back(1);
        }
        else if (global_row == 1) {
            values.push_back(1.0);
            col_indices.push_back(0);

            values.push_back(3.0);
            col_indices.push_back(1);
        }

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

    Vector local_b_values(local_rows, 0.0);
    Vector local_x0_values(local_rows, 0.0);

    for (Index local_row = 0; local_row < local_rows; ++local_row) {
        const Index global_row = local_start + local_row;

        if (global_row == 0) {
            local_b_values[local_row] = 6.0;
        }
        else if (global_row == 1) {
            local_b_values[local_row] = 7.0;
        }
    }

    DistributedVector b(
        global_size,
        local_start,
        local_b_values,
        MPI_COMM_WORLD
    );

    DistributedVector x0(
        global_size,
        local_start,
        local_x0_values,
        MPI_COMM_WORLD
    );

    GMRESConfig config;
    config.restart_blocks = 2;
    config.s_step = 1;
    config.max_iterations = 10;
    config.tolerance = 1e-10;
    config.verbose = rank == 0;

    CAGMRESMPIResult result = gmres_ca_mpi(A, b, x0, config);

    assert(result.converged);

    const Vector& local_x = result.x.local_values();

    for (Index local_row = 0; local_row < local_rows; ++local_row) {
        const Index global_row = local_start + local_row;

        if (global_row == 0) {
            assert(nearly_equal(local_x[local_row], 1.0, 1e-8));
        }
        else if (global_row == 1) {
            assert(nearly_equal(local_x[local_row], 2.0, 1e-8));
        }
    }

    DistributedVector Ax = A.multiply(result.x);
    DistributedVector residual = b;
    axpy_local(-1.0, Ax, residual);

    assert(norm2_mpi(residual) < 1e-8);

    if (rank == 0) {
        std::println("test_gmres_ca_mpi passed.");
        std::println("iterations = {}", result.iterations);
        std::println("blocks completed = {}", result.blocks_completed);
    }

    MPI_Finalize();

    return 0;
}