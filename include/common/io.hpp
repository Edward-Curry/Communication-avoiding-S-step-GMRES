/**
 * @file include/common/io.hpp
 * @brief Declares Matrix Market and experiment-output I/O.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

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
    /**
     * @brief Records one residual-history observation.
     */
    struct ResidualHistoryEntry
    {
        Index history_entry = 0;
        Index iteration = 0;
        std::string kind;
        Scalar residual_norm = 0.0;
        /// @brief Accepted CA block width, or zero when not applicable.
        Index block_s = 0;
    };

    /**
     * @brief Summarises cached halo exchanges over MPI ranks.
     */
    struct HaloExchangeSummary
    {
        bool available = false;
        double mean_receive_peers = 0.0;
        Index max_receive_peers = 0;
        double mean_send_peers = 0.0;
        Index max_send_peers = 0;
        double mean_receive_values = 0.0;
        Index max_receive_values = 0;
        double mean_send_values = 0.0;
        Index max_send_values = 0;
    };

    /**
     * @brief Collects inputs and measurements written by an experiment.
     */
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
        HaloExchangeSummary halo_exchange;

        bool converged = false;
        Index iterations = 0;
        Index blocks_completed = 0;

        std::vector<ResidualHistoryEntry> residual_history;
        Vector solution;
        Vector exact_solution;

        /// @brief States whether exact_solution supports a forward-error check.
        bool has_exact_solution = true;
        /// @brief Controls whether the final solution vector is written.
        bool write_solution_output = true;

        Scalar initial_residual = 0.0;
        Scalar final_residual = 0.0;
        Scalar relative_residual = 0.0;
        Scalar relative_forward_error = 0.0;
        double solve_seconds = 0.0;
    };

    /**
     * @brief Reads a sparse Matrix Market coordinate matrix.
     * @param filename Matrix Market file path.
     * @return Matrix stored in CSR format.
     */
    SparseMatrixCSR read_matrix_market(const std::string& filename);

    /**
     * @brief Reads a Matrix Market array vector.
     * @param filename Matrix Market vector file path.
     * @return Dense vector read from the file.
     */
    Vector read_matrix_market_vector(const std::string& filename);

    /**
     * @brief Writes a plain residual sequence to CSV.
     * @param filename Output CSV path.
     * @param residuals Residual values to write.
     */
    void write_residual_history(const std::string& filename,
                                const Vector& residuals);

    /**
     * @brief Writes a dense vector to one-column CSV.
     * @param filename Output CSV path.
     * @param x Vector to write.
     */
    void write_vector_csv(const std::string& filename,
                          const Vector& x);

    /**
     * @brief Converts cycle residuals into iteration-indexed observations.
     * @param residuals One residual value per recorded step.
     * @param iterations Total iteration count.
     * @param restart Restart length used by the solver.
     * @return Residual-history entries with restart-boundary labels.
     */
    std::vector<ResidualHistoryEntry> make_restarted_residual_history(
        const Vector& residuals,
        Index iterations,
        Index restart);

    /**
     * @brief Writes performance, accuracy, convergence, and history outputs.
     * @param output_directory Destination directory.
     * @param experiment Complete experiment description and measurements.
     */
    void write_gmres_experiment_outputs(
        const std::filesystem::path& output_directory,
        const GMRESExperimentData& experiment);
}

#endif
