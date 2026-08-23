/**
 * @file tests/test_distributed_sparse_matrix.cpp
 * @brief Tests distributed sparse matrix multiplication and halo statistics.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "parallel/distributed_sparse_matrix.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <print>
#include <vector>

namespace
{
    /**
     * @brief Returns the contiguous local size assigned to an MPI rank.
     * @param n Global vector length.
     * @param rank MPI rank.
     * @param size Number of MPI ranks.
     * @return Number of locally owned entries.
     */
    gmres::Index local_size(gmres::Index n, int rank, int size)
    {
        return n / size + (static_cast<gmres::Index>(rank) < n % size ? 1 : 0);
    }

    /**
     * @brief Returns the first global index assigned to an MPI rank.
     * @param n Global vector length.
     * @param rank MPI rank.
     * @param size Number of MPI ranks.
     * @return First locally owned global index.
     */
    gmres::Index local_start(gmres::Index n, int rank, int size)
    {
        return static_cast<gmres::Index>(rank) * (n / size) +
               std::min(static_cast<gmres::Index>(rank), n % size);
    }
}

/**
 * @brief Runs the distributed sparse matrix test.
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

    constexpr gmres::Index n = 8;
    const gmres::Index start = local_start(n, rank, size);
    const gmres::Index rows = local_size(n, rank, size);

    gmres::Vector values;
    std::vector<gmres::Index> columns;
    std::vector<gmres::Index> row_ptr{0};

    // Tridiagonal structure creates halo dependencies at rank boundaries.
    for (gmres::Index local_row = 0; local_row < rows; ++local_row)
    {
        const gmres::Index row = start + local_row;
        if (row > 0)
        {
            values.push_back(-1.0);
            columns.push_back(row - 1);
        }
        values.push_back(2.0);
        columns.push_back(row);
        if (row + 1 < n)
        {
            values.push_back(-1.0);
            columns.push_back(row + 1);
        }
        row_ptr.push_back(values.size());
    }

    gmres::DistributedSparseMatrixCSR matrix(n, n, start, rows,
                                              values, columns, row_ptr,
                                              MPI_COMM_WORLD);

    gmres::Vector local_x(rows, 0.0);
    for (gmres::Index i = 0; i < rows; ++i)
    {
        local_x[i] = static_cast<double>(start + i + 1);
    }
    gmres::DistributedVector x(n, start, local_x, MPI_COMM_WORLD);

    // The second SpMV reuses the cached halo plan.
    for (int repetition = 0; repetition < 2; ++repetition)
    {
        const gmres::DistributedVector y = matrix.multiply(x);
        for (gmres::Index i = 0; i < rows; ++i)
        {
            const gmres::Index row = start + i;
            const double expected = row == n - 1 ? 9.0 : 0.0;
            assert(std::abs(y.local_values()[i] - expected) < 1e-12);
        }
    }

    const gmres::HaloExchangeStatistics halo =
        matrix.halo_exchange_statistics();
    const gmres::Index expected_halo_values =
        (start > 0 ? 1 : 0) + (start + rows < n ? 1 : 0);

    assert(halo.receive_peer_count == expected_halo_values);
    assert(halo.send_peer_count == expected_halo_values);
    assert(halo.receive_value_count == expected_halo_values);
    assert(halo.send_value_count == expected_halo_values);

    if (rank == 0)
    {
        std::println("Distributed sparse matrix halo test passed with {} ranks.", size);
    }

    MPI_Finalize();
    return 0;
}
