#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <print>
#include <utility>
#include <vector>

namespace
{
    gmres::Index local_size_for_rank(gmres::Index global_size,
                                     int rank,
                                     int comm_size)
    {
        const gmres::Index base =
            global_size / static_cast<gmres::Index>(comm_size);
        const gmres::Index remainder =
            global_size % static_cast<gmres::Index>(comm_size);

        return base +
            (static_cast<gmres::Index>(rank) < remainder ? 1 : 0);
    }

    gmres::Index local_start_for_rank(gmres::Index global_size,
                                      int rank,
                                      int comm_size)
    {
        const gmres::Index base =
            global_size / static_cast<gmres::Index>(comm_size);
        const gmres::Index remainder =
            global_size % static_cast<gmres::Index>(comm_size);
        const gmres::Index rank_index = static_cast<gmres::Index>(rank);

        return rank_index * base + std::min(rank_index, remainder);
    }

    gmres::Scalar input_value(gmres::Index index, gmres::Scalar scale)
    {
        return scale * static_cast<gmres::Scalar>(index + 1);
    }

    gmres::Scalar expected_value(gmres::Index row,
                                 gmres::Index global_size,
                                 gmres::Scalar scale)
    {
        gmres::Scalar value = 2.0 * input_value(row, scale);

        if (row > 0)
        {
            value -= input_value(row - 1, scale);
        }

        if (row + 1 < global_size)
        {
            value += 0.5 * input_value(row + 1, scale);
        }

        return value;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int comm_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    const gmres::Index global_size =
        std::max<gmres::Index>(4, static_cast<gmres::Index>(comm_size) * 2);
    const gmres::Index local_start =
        local_start_for_rank(global_size, rank, comm_size);
    const gmres::Index local_rows =
        local_size_for_rank(global_size, rank, comm_size);

    gmres::Vector values;
    std::vector<gmres::Index> column_indices;
    std::vector<gmres::Index> row_ptr{0};

    for (gmres::Index local_row = 0; local_row < local_rows; ++local_row)
    {
        const gmres::Index global_row = local_start + local_row;

        if (global_row > 0)
        {
            values.push_back(-1.0);
            column_indices.push_back(global_row - 1);
        }

        values.push_back(2.0);
        column_indices.push_back(global_row);

        if (global_row + 1 < global_size)
        {
            values.push_back(0.5);
            column_indices.push_back(global_row + 1);
        }

        row_ptr.push_back(values.size());
    }

    gmres::DistributedSparseMatrixCSR matrix(global_size,
                                              global_size,
                                              local_start,
                                              local_rows,
                                              values,
                                              column_indices,
                                              row_ptr,
                                              MPI_COMM_WORLD);

    for (gmres::Scalar scale : {1.0, -0.5})
    {
        gmres::Vector local_x(local_rows, 0.0);

        for (gmres::Index i = 0; i < local_rows; ++i)
        {
            local_x[i] = input_value(local_start + i, scale);
        }

        gmres::DistributedVector x(global_size,
                                   local_start,
                                   std::move(local_x),
                                   MPI_COMM_WORLD);
        const gmres::DistributedVector y = matrix.multiply(x);

        for (gmres::Index i = 0; i < local_rows; ++i)
        {
            const gmres::Scalar expected =
                expected_value(local_start + i, global_size, scale);
            assert(std::abs(y.local_values()[i] - expected) < 1e-12);
        }
    }

    if (rank == 0)
    {
        std::println("Distributed sparse matrix halo exchange test passed.");
    }

    MPI_Finalize();
    return 0;
}
