#include "communication_avoiding/gmres_ca_mpi.hpp"

#include "common/config.hpp"
#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"
#include "parallel/distributed_vector_ops.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <mpi.h>
#include <print>
#include <vector>

namespace
{
    gmres::Index local_size_for_rank(gmres::Index global_size,
                                     int rank,
                                     int comm_size)
    {
        gmres::Index base = global_size / static_cast<gmres::Index>(comm_size);
        gmres::Index remainder = global_size % static_cast<gmres::Index>(comm_size);

        return base + (static_cast<gmres::Index>(rank) < remainder ? 1 : 0);
    }

    gmres::Index local_start_for_rank(gmres::Index global_size,
                                      int rank,
                                      int comm_size)
    {
        gmres::Index base = global_size / static_cast<gmres::Index>(comm_size);
        gmres::Index remainder = global_size % static_cast<gmres::Index>(comm_size);
        gmres::Index rank_index = static_cast<gmres::Index>(rank);

        return rank_index * base + std::min(rank_index, remainder);
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;

    int rank = 0;
    int comm_size = 0;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);

    const gmres::Index global_size = 2;

    gmres::Index local_start =
        local_start_for_rank(global_size, rank, comm_size);

    gmres::Index local_rows =
        local_size_for_rank(global_size, rank, comm_size);

    // Global matrix:
    //
    // A = [ 4  1
    //       1  3 ]
    //
    // Exact solution:
    //
    // x = [1, 2]
    //
    // b = A*x = [6, 7]

    gmres::Vector values;
    std::vector<gmres::Index> col_indices;
    std::vector<gmres::Index> row_ptr;

    row_ptr.push_back(0);

    for (gmres::Index local_row = 0; local_row < local_rows; ++local_row)
    {
        gmres::Index global_row = local_start + local_row;

        if (global_row == 0)
        {
            values.push_back(4.0);
            col_indices.push_back(0);

            values.push_back(1.0);
            col_indices.push_back(1);
        }
        else if (global_row == 1)
        {
            values.push_back(1.0);
            col_indices.push_back(0);

            values.push_back(3.0);
            col_indices.push_back(1);
        }

        row_ptr.push_back(values.size());
    }

    gmres::DistributedSparseMatrixCSR A(global_size,
                                        global_size,
                                        local_start,
                                        local_rows,
                                        values,
                                        col_indices,
                                        row_ptr,
                                        comm);

    gmres::Vector b_local(local_rows, 0.0);
    gmres::Vector x0_local(local_rows, 0.0);

    for (gmres::Index i = 0; i < local_rows; ++i)
    {
        gmres::Index global_index = local_start + i;

        if (global_index == 0)
        {
            b_local[i] = 6.0;
        }
        else if (global_index == 1)
        {
            b_local[i] = 7.0;
        }
    }

    gmres::DistributedVector b(global_size,
                               local_start,
                               b_local,
                               comm);

    gmres::DistributedVector x0(global_size,
                                local_start,
                                x0_local,
                                comm);

    gmres::GMRESConfig config;
    config.restart_blocks = 2;
    config.s_step = 1;
    config.max_iterations = 10;
    config.tolerance = 1e-10;
    config.verbose = false;

    gmres::CAGMRESMPIResult result = gmres::gmres_ca_mpi(A, b, x0, config);

    assert(result.converged);
    assert(result.iterations <= config.max_iterations);

    const gmres::Vector& x_local = result.x.local_values();

    for (gmres::Index i = 0; i < result.x.local_size(); ++i)
    {
        gmres::Index global_index = result.x.local_start() + i;

        if (global_index == 0)
        {
            assert(std::abs(x_local[i] - 1.0) < 1e-8);
        }
        else if (global_index == 1)
        {
            assert(std::abs(x_local[i] - 2.0) < 1e-8);
        }
    }

    gmres::DistributedVector Ax = A.multiply(result.x);
    gmres::DistributedVector residual = b;

    gmres::axpy_local(-1.0, Ax, residual);

    assert(gmres::norm2_mpi(residual) < config.tolerance);

    // Block recycling smoke test: restart_blocks=1, s_step=1 forces several
    // restart cycles on this 2-dimensional system, giving a recycled subspace
    // from one cycle's winning block a real chance to seed the next.
    gmres::GMRESConfig recycling_config = config;
    recycling_config.restart_blocks = 1;
    recycling_config.s_step = 1;
    recycling_config.max_iterations = 20;
    recycling_config.enable_recycling = true;
    recycling_config.recycle_count = 1;
    // Isolate recycling from adaptive-s: on this tiny 2-dimensional system,
    // s_initial_probe's s_max request can exactly exhaust the remaining
    // dimensions right as a recycled subspace is seeded, an unrelated
    // numerical edge case rather than anything specific to recycling.
    recycling_config.adaptive_s = false;
    recycling_config.s_initial_probe = false;

    gmres::CAGMRESMPIResult recycled = gmres::gmres_ca_mpi(A, b, x0, recycling_config);

    assert(recycled.converged);

    gmres::DistributedVector Ax_recycled = A.multiply(recycled.x);
    gmres::DistributedVector residual_recycled = b;

    gmres::axpy_local(-1.0, Ax_recycled, residual_recycled);

    assert(gmres::norm2_mpi(residual_recycled) < recycling_config.tolerance);

    if (rank == 0)
    {
        std::println("MPI CA-GMRES test passed.");
    }

    MPI_Finalize();

    return 0;
}