/**
 * @file experiments/run_gmres_ca_mpi_experiment.cpp
 * @brief Runs a distributed communication-avoiding GMRES experiment.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "common/config.hpp"
#include "common/io.hpp"
#include "communication_avoiding/gmres_ca_mpi.hpp"
#include "parallel/distributed_io.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_utils.hpp"
#include "parallel/distributed_vector.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <mpi.h>
#include <print>
#include <stdexcept>
#include <string>

namespace
{
    /**
     * @brief Converts CA-GMRES residual samples to common output records.
     * @param samples Residual samples reported by the CA-GMRES solver.
     * @return Residual records suitable for CSV output.
     */
    std::vector<gmres::ResidualHistoryEntry> make_ca_residual_history(
        const gmres::CAResidualHistory& samples)
    {
        std::vector<gmres::ResidualHistoryEntry> history;
        history.reserve(samples.size());

        for (gmres::Index i = 0; i < samples.size(); ++i)
        {
            const gmres::CAResidualSample& sample = samples[i];

            std::string kind = "block_estimated";

            if (i == 0)
            {
                kind = "initial";
            }
            else if (sample.from_recycle_seed)
            {
                kind = "recycle_seeded";
            }
            else if (sample.recomputed)
            {
                kind = "restart_recomputed";
            }

            history.push_back({
                i,
                sample.iteration,
                kind,
                sample.residual_norm,
                sample.block_s
            });
        }

        return history;
    }
}

/**
 * @brief Runs the MPI CA-GMRES experiment.
 * @param argc Number of command-line arguments.
 * @param argv Matrix, output directory, and optional vector-file arguments.
 * @return Collective status code for the MPI job.
 */
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
        const std::string rhs_path = argc > 3 ? argv[3] : "";
        const std::string exact_solution_path = argc > 4 ? argv[4] : "";
        const bool has_rhs_file = !rhs_path.empty();
        const bool has_exact_solution_file = !exact_solution_path.empty();
        const bool has_exact_solution = has_exact_solution_file || !has_rhs_file;
        const bool write_solution_output = argc <= 5;

        if (argc > 6
            || (argc == 6
                && std::string(argv[5]) != "--no-solution-output"))
        {
            throw std::invalid_argument(
                "Usage: run_gmres_ca_mpi_experiment <matrix> <output> [rhs] [exact_solution] [--no-solution-output]");
        }

        const gmres::DistributedSparseMatrixCSR A =
            gmres::read_matrix_market_distributed(
                matrix_path,
                MPI_COMM_WORLD);

        if (A.global_rows() != A.global_cols())
        {
            throw std::runtime_error("MPI CA-GMRES requires a square matrix.");
        }

        const gmres::Vector local_initial_values(A.local_rows(), 0.0);
        const gmres::DistributedVector x0(
            A.global_cols(),
            A.local_row_start(),
            local_initial_values,
            MPI_COMM_WORLD);

        gmres::DistributedVector x_true;

        if (has_exact_solution)
        {
            if (has_exact_solution_file)
            {
                x_true = gmres::read_matrix_market_vector_distributed(
                    exact_solution_path,
                    MPI_COMM_WORLD);

                if (x_true.global_size() != A.global_cols())
                {
                    throw std::runtime_error(
                        "Exact solution size must match matrix columns.");
                }
            }
            else
            {
                const gmres::Vector local_true_values(A.local_rows(), 1.0);
                x_true = gmres::DistributedVector(
                    A.global_cols(),
                    A.local_row_start(),
                    local_true_values,
                    MPI_COMM_WORLD);
            }
        }

        gmres::DistributedVector b;

        if (has_rhs_file)
        {
            b = gmres::read_matrix_market_vector_distributed(
                rhs_path,
                MPI_COMM_WORLD);

            if (b.global_size() != A.global_rows())
            {
                throw std::runtime_error(
                    "Right-hand side size must match matrix rows.");
            }
        }
        else
        {
            b = A.multiply(x_true);
        }

        gmres::GMRESConfig config;

        /// Allows recycle-count sweeps without rebuilding the executable.
        if (const char* const requested = std::getenv("GMRES_RECYCLE_COUNT"))
        {
            const std::string text(requested);
            std::size_t consumed = 0;
            long long value = 0;
            try
            {
                value = std::stoll(text, &consumed);
            }
            catch (const std::exception&)
            {
                consumed = 0;
            }

            if (consumed != text.size() || value < 0)
            {
                throw std::invalid_argument(
                    "GMRES_RECYCLE_COUNT must be a non-negative integer, got: "
                    + text);
            }

            config.recycle_count = static_cast<gmres::Index>(value);

            /// A zero recycle count disables recycling.
            config.enable_recycling = value > 0;

            if (rank == 0)
            {
                std::println("Recycle count overridden: {} (deflation {})",
                             config.recycle_count,
                             config.enable_recycling ? "on" : "off");
            }
        }

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
        const gmres::Scalar forward_error = has_exact_solution
            ? gmres::relative_forward_error_mpi(result.x, x_true)
            : 0.0;

        const gmres::Vector solution = write_solution_output
            ? gmres::gather_distributed_vector(result.x, MPI_COMM_WORLD)
            : gmres::Vector();
        const gmres::Vector exact_solution =
            write_solution_output && has_exact_solution
                ? gmres::gather_distributed_vector(x_true, MPI_COMM_WORLD)
                : gmres::Vector();

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

        const gmres::HaloExchangeSummary halo_exchange =
            gmres::halo_exchange_summary_mpi(A);

        if (rank == 0)
        {
            gmres::GMRESExperimentData experiment;
            experiment.matrix_path = matrix_path;
            experiment.solver_name = "mpi_ca_gmres";
            experiment.output_prefix = "gmres_ca_mpi";
            experiment.initial_guess_description = "all zeros";
            experiment.has_exact_solution = has_exact_solution;
            experiment.exact_solution_description = has_exact_solution_file
                ? ("loaded from " + exact_solution_path)
                : (has_exact_solution
                       ? "all ones"
                       : "unknown (right-hand side supplied directly)");
            experiment.right_hand_side_description = has_rhs_file
                ? ("b loaded from " + rhs_path)
                : "b = A * exact_solution";
            experiment.rows = A.global_rows();
            experiment.cols = A.global_cols();
            experiment.nonzeros =
                static_cast<gmres::Index>(global_nonzeros);
            experiment.process_count =
                static_cast<gmres::Index>(process_count);
            experiment.config = config;
            experiment.halo_exchange = halo_exchange;
            experiment.converged = result.converged;
            experiment.iterations = result.iterations;
            experiment.blocks_completed = result.blocks_completed;
            experiment.residual_history =
                make_ca_residual_history(result.residual_history);
            experiment.solution = solution;
            experiment.exact_solution = exact_solution;
            experiment.initial_residual = initial_residual;
            experiment.final_residual = final_residual;
            experiment.relative_residual = relative_residual;
            experiment.relative_forward_error = forward_error;
            experiment.solve_seconds = solve_seconds;
            experiment.write_solution_output = write_solution_output;

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
