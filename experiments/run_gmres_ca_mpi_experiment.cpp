#include "common/config.hpp"
#include "common/io.hpp"
#include "communication_avoiding/gmres_ca_mpi.hpp"
#include "parallel/distributed_io.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_utils.hpp"
#include "parallel/distributed_vector.hpp"

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <mpi.h>
#include <print>
#include <stdexcept>
#include <string>

namespace
{
    std::vector<gmres::ResidualHistoryEntry> make_ca_residual_history(
        const gmres::Vector& residuals,
        gmres::Index iterations,
        const gmres::GMRESConfig& config)
    {
        std::vector<gmres::ResidualHistoryEntry> history;
        history.reserve(residuals.size());

        if (residuals.empty())
        {
            return history;
        }

        gmres::Index history_entry = 0;
        gmres::Index iteration = 0;
        const gmres::Index cycle_capacity =
            config.restart_blocks * config.s_step;

        history.push_back({
            history_entry,
            iteration,
            "initial",
            residuals[history_entry]
        });
        ++history_entry;

        while (history_entry < residuals.size())
        {
            iteration = std::min(iterations, iteration + cycle_capacity);

            history.push_back({
                history_entry,
                iteration,
                "block_cycle_estimated",
                residuals[history_entry]
            });
            ++history_entry;

            if (history_entry < residuals.size())
            {
                history.push_back({
                    history_entry,
                    iteration,
                    "restart_recomputed",
                    residuals[history_entry]
                });
                ++history_entry;
            }
        }

        return history;
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    int process_count = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &process_count);

    int local_status = 0;

    try
    {
        const std::string matrix_path =
            argc > 1 ? argv[1] : "data/matrices/test_3x3.mtx";
        const std::filesystem::path output_directory =
            argc > 2 ? argv[2] : "data/outputs";

        const gmres::DistributedSparseMatrixCSR A =
            gmres::read_matrix_market_distributed(
                matrix_path,
                MPI_COMM_WORLD);

        if (A.global_rows() != A.global_cols())
        {
            throw std::runtime_error("MPI CA-GMRES requires a square matrix.");
        }

        const gmres::Vector local_true_values(A.local_rows(), 1.0);
        const gmres::Vector local_initial_values(A.local_rows(), 0.0);
        const gmres::DistributedVector x_true(
            A.global_cols(),
            A.local_row_start(),
            local_true_values,
            MPI_COMM_WORLD);
        const gmres::DistributedVector x0(
            A.global_cols(),
            A.local_row_start(),
            local_initial_values,
            MPI_COMM_WORLD);
        const gmres::DistributedVector b = A.multiply(x_true);

        gmres::GMRESConfig config;
        config.restart_blocks = 6;
        config.s_step =
            std::min<gmres::Index>(5, A.global_rows());
        config.max_iterations = 200;
        config.tolerance = 1e-10;
        config.verbose = false;

        const gmres::Scalar initial_residual =
            gmres::residual_norm_mpi(A, b, x0);

        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        const gmres::CAGMRESMPIResult result =
            gmres::gmres_ca_mpi(A, b, x0, config);
        MPI_Barrier(MPI_COMM_WORLD);
        const double local_solve_seconds = MPI_Wtime() - start;

        double solve_seconds = 0.0;
        MPI_Reduce(&local_solve_seconds,
                   &solve_seconds,
                   1,
                   MPI_DOUBLE,
                   MPI_MAX,
                   0,
                   MPI_COMM_WORLD);

        const gmres::Scalar final_residual =
            gmres::residual_norm_mpi(A, b, result.x);
        const gmres::Scalar relative_residual =
            gmres::relative_residual_norm_mpi(A, b, result.x);
        const gmres::Scalar forward_error =
            gmres::relative_forward_error_mpi(result.x, x_true);

        const gmres::Vector solution =
            gmres::gather_distributed_vector(
                result.x,
                MPI_COMM_WORLD);
        const gmres::Vector exact_solution =
            gmres::gather_distributed_vector(
                x_true,
                MPI_COMM_WORLD);

        const unsigned long long local_nonzeros =
            static_cast<unsigned long long>(A.nonzeros());
        unsigned long long global_nonzeros = 0;
        MPI_Reduce(&local_nonzeros,
                   &global_nonzeros,
                   1,
                   MPI_UNSIGNED_LONG_LONG,
                   MPI_SUM,
                   0,
                   MPI_COMM_WORLD);

        if (rank == 0)
        {
            gmres::GMRESExperimentData experiment;
            experiment.matrix_path = matrix_path;
            experiment.solver_name = "mpi_ca_gmres";
            experiment.output_prefix = "gmres_ca_mpi";
            experiment.initial_guess_description = "all zeros";
            experiment.exact_solution_description = "all ones";
            experiment.right_hand_side_description =
                "b = A * exact_solution";
            experiment.rows = A.global_rows();
            experiment.cols = A.global_cols();
            experiment.nonzeros =
                static_cast<gmres::Index>(global_nonzeros);
            experiment.process_count =
                static_cast<gmres::Index>(process_count);
            experiment.config = config;
            experiment.converged = result.converged;
            experiment.iterations = result.iterations;
            experiment.blocks_completed = result.blocks_completed;
            experiment.residual_history =
                make_ca_residual_history(
                    result.residual_history,
                    result.iterations,
                    config);
            experiment.solution = solution;
            experiment.exact_solution = exact_solution;
            experiment.initial_residual = initial_residual;
            experiment.final_residual = final_residual;
            experiment.relative_residual = relative_residual;
            experiment.relative_forward_error = forward_error;
            experiment.solve_seconds = solve_seconds;

            gmres::write_gmres_experiment_outputs(
                output_directory,
                experiment);
        }
    }
    catch (const std::exception& error)
    {
        if (rank == 0)
        {
            std::println(
                stderr,
                "MPI CA-GMRES experiment failed: {}",
                error.what());
        }

        local_status = 1;
    }

    int global_status = 0;
    MPI_Allreduce(&local_status,
                  &global_status,
                  1,
                  MPI_INT,
                  MPI_MAX,
                  MPI_COMM_WORLD);

    MPI_Finalize();
    return global_status;
}
