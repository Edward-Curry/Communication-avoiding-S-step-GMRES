#include "common/config.hpp"
#include "common/io.hpp"
#include "common/sparse_matrix.hpp"
#include "common/utils.hpp"
#include "communication_avoiding/gmres_ca.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <exception>
#include <filesystem>
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
    const std::string matrix_path =
        argc > 1 ? argv[1] : "data/matrices/test_3x3.mtx";
    const std::filesystem::path output_directory =
        argc > 2 ? argv[2] : "data/outputs";

    try
    {
        const gmres::SparseMatrixCSR A =
            gmres::read_matrix_market(matrix_path);

        if (A.rows() != A.cols())
        {
            throw std::runtime_error("Sequential CA-GMRES requires a square matrix.");
        }

        const gmres::Vector x_true(A.cols(), 1.0);
        const gmres::Vector b = A.multiply(x_true);
        const gmres::Vector x0(A.cols(), 0.0);

        gmres::GMRESConfig config;

        const gmres::Scalar initial_residual =
            gmres::residual_norm(A, b, x0);

        const auto start = std::chrono::steady_clock::now();
        const gmres::CAGMRESResult result =
            gmres::gmres_ca(A, b, x0, config);
        const auto end = std::chrono::steady_clock::now();

        const double solve_seconds =
            std::chrono::duration<double>(end - start).count();

        const gmres::Scalar final_residual =
            gmres::residual_norm(A, b, result.x);
        const gmres::Scalar relative_residual =
            gmres::relative_residual_norm(A, b, result.x);
        const gmres::Scalar forward_error =
            gmres::relative_forward_error(result.x, x_true);

        gmres::GMRESExperimentData experiment;
        experiment.matrix_path = matrix_path;
        experiment.solver_name = "sequential_ca_gmres";
        experiment.output_prefix = "gmres_ca";
        experiment.initial_guess_description = "all zeros";
        experiment.exact_solution_description = "all ones";
        experiment.right_hand_side_description =
            "b = A * exact_solution";
        experiment.rows = A.rows();
        experiment.cols = A.cols();
        experiment.nonzeros = A.nonzeros();
        experiment.config = config;
        experiment.converged = result.converged;
        experiment.iterations = result.iterations;
        experiment.blocks_completed = result.blocks_completed;
        experiment.residual_history =
            make_ca_residual_history(
                result.residual_history,
                result.iterations,
                config);
        experiment.solution = result.x;
        experiment.exact_solution = x_true;
        experiment.initial_residual = initial_residual;
        experiment.final_residual = final_residual;
        experiment.relative_residual = relative_residual;
        experiment.relative_forward_error = forward_error;
        experiment.solve_seconds = solve_seconds;

        gmres::write_gmres_experiment_outputs(
            output_directory,
            experiment);
    }
    catch (const std::exception& error)
    {
        std::println(stderr, "CA-GMRES experiment failed: {}", error.what());
        return 1;
    }

    return 0;
}
