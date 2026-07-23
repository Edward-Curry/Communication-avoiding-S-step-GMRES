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
            make_ca_residual_history(result.residual_history);
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
