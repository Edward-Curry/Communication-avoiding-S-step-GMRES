#ifndef COMMON_IO_HPP
#define COMMON_IO_HPP

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace gmres
{
    struct ResidualHistoryEntry
    {
        Index history_entry = 0;
        Index iteration = 0;
        std::string kind;
        Scalar residual_norm = 0.0;
        // s-step block width for this entry (CA solvers only); 0 when not
        // applicable (initial/restart rows and non-CA solvers).
        Index block_s = 0;
    };

    struct GMRESExperimentData
    {
        std::string matrix_path;
        std::string solver_name;
        std::string output_prefix;
        std::string initial_guess_description;
        std::string exact_solution_description;
        std::string right_hand_side_description;

        Index rows = 0;
        Index cols = 0;
        Index nonzeros = 0;
        Index process_count = 1;
        GMRESConfig config;

        bool converged = false;
        Index iterations = 0;
        Index blocks_completed = 0;

        std::vector<ResidualHistoryEntry> residual_history;
        Vector solution;
        Vector exact_solution;

        Scalar initial_residual = 0.0;
        Scalar final_residual = 0.0;
        Scalar relative_residual = 0.0;
        Scalar relative_forward_error = 0.0;
        double solve_seconds = 0.0;
    };

    SparseMatrixCSR read_matrix_market(const std::string& filename);

    Vector read_matrix_market_vector(const std::string& filename);

    void write_residual_history(const std::string& filename,
                                const Vector& residuals);

    void write_vector_csv(const std::string& filename,
                          const Vector& x);

    std::vector<ResidualHistoryEntry> make_restarted_residual_history(
        const Vector& residuals,
        Index iterations,
        Index restart);

    void write_gmres_experiment_outputs(
        const std::filesystem::path& output_directory,
        const GMRESExperimentData& experiment);
}

#endif
