/**
 * @file src/common/io.cpp
 * @brief Implements Matrix Market input and experiment CSV output.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "common/io.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gmres
{
    namespace
    {
        /**
         * @brief Opens a CSV file with full floating-point precision.
         * @param path Destination file path.
         * @return Open output stream.
         */
        std::ofstream open_csv(const std::filesystem::path& path)
        {
            std::ofstream output(path);

            if (!output)
            {
                throw std::runtime_error(
                    "Could not open output file: " + path.string());
            }

            output << std::setprecision(
                std::numeric_limits<double>::max_digits10);
            return output;
        }

        /**
         * @brief Converts a value to a filesystem-safe filename component.
         * @param value Source value.
         * @return Sanitised filename component.
         */
        std::string make_filename_component(std::string value)
        {
            for (char& character : value)
            {
                const unsigned char byte =
                    static_cast<unsigned char>(character);

                if (!std::isalnum(byte)
                    && character != '-'
                    && character != '_')
                {
                    character = '_';
                }
            }

            const auto first = value.find_first_not_of('_');
            const auto last = value.find_last_not_of('_');

            if (first == std::string::npos)
            {
                return "matrix";
            }

            return value.substr(first, last - first + 1);
        }

        /**
         * @brief Converts a block-orthogonalization method to CSV text.
         * @param method Method to convert.
         * @return Stable CSV label.
         */
        const char* to_csv(BlockOrthogonalizationMethod method)
        {
            switch (method)
            {
            case BlockOrthogonalizationMethod::ModifiedGramSchmidt:
                return "modified_gram_schmidt";
            case BlockOrthogonalizationMethod::BCGS2CholQR:
                return "bcgs2_cholqr";
            }

            return "unknown";
        }

        /**
         * @brief Converts a partial-Cholesky stopping rule to CSV text.
         * @param rule Stopping rule to convert.
         * @return Stable CSV label.
         */
        const char* to_csv(PartialCholeskyStoppingRule rule)
        {
            switch (rule)
            {
            case PartialCholeskyStoppingRule::PivotOnly:
                return "pivot_only";
            case PartialCholeskyStoppingRule::TriangularConditionEstimate:
                return "triangular_condition_estimate";
            }

            return "unknown";
        }

        /**
         * @brief Converts a polynomial basis type to CSV text.
         * @param basis Basis type to convert.
         * @return Stable CSV label.
         */
        const char* to_csv(PolynomialBasisType basis)
        {
            switch (basis)
            {
            case PolynomialBasisType::Monomial:
                return "monomial";
            case PolynomialBasisType::Newton:
                return "newton";
            case PolynomialBasisType::ScaledNewton:
                return "scaled_newton";
            }

            return "unknown";
        }

        /**
         * @brief Converts a string to lower case.
         * @param value String to convert.
         * @return Lower-case copy of value.
         */
        std::string lowercase(std::string value)
        {
            std::transform(
                value.begin(),
                value.end(),
                value.begin(),
                [](unsigned char character)
                {
                    return static_cast<char>(std::tolower(character));
                });

            return value;
        }

        /**
         * @brief Reads the next non-comment Matrix Market line.
         * @param file Open Matrix Market file.
         * @param filename File name used in error messages.
         * @return First nonempty, non-comment line.
         */
        std::string read_matrix_market_data_line(
            std::ifstream& file,
            const std::string& filename)
        {
            std::string line;

            do
            {
                if (!std::getline(file, line))
                {
                    throw std::runtime_error(
                        "Missing Matrix Market size line in file: "
                        + filename);
                }
            } while (line.empty() || line[0] == '%');

            return line;
        }

        /**
         * @brief Writes CSV metadata column names.
         * @param output Open CSV stream.
         */
        void write_common_header(std::ofstream& output)
        {
            output
                << "matrix,solver,rows,cols,nonzeros,process_count,"
                   "restart,restart_blocks,s_step,max_iterations,"
                   "relative_tolerance";
        }

        /**
         * @brief Writes CSV metadata values for one experiment.
         * @param output Open CSV stream.
         * @param experiment Experiment metadata.
         */
        void write_common_values(std::ofstream& output,
                                 const GMRESExperimentData& experiment)
        {
            output
                << std::quoted(experiment.matrix_path) << ','
                << experiment.solver_name << ','
                << experiment.rows << ','
                << experiment.cols << ','
                << experiment.nonzeros << ','
                << experiment.process_count << ','
                << experiment.config.restart << ','
                << experiment.config.restart_blocks << ','
                << experiment.config.s_step << ','
                << experiment.config.max_iterations << ','
                << experiment.config.tolerance;
        }

        /**
         * @brief Writes the experiment configuration CSV file.
         * @param path Destination CSV path.
         * @param experiment Experiment metadata and configuration.
         */
        void write_experiment_config(
            const std::filesystem::path& path,
            const GMRESExperimentData& experiment)
        {
            std::ofstream output = open_csv(path);

            write_common_header(output);
            output
                << ",block_orthogonalization,partial_cholesky_stopping_rule,"
                   "partial_cholesky_condition_limit,adaptive_s,s_min,s_max,"
                   "s_grow_after,s_initial_probe,polynomial_basis,enable_recycling,"
                   "recycle_count,"
                   "verbose,initial_guess,has_exact_solution,exact_solution,"
                   "right_hand_side,write_solution_output\n";

            write_common_values(output, experiment);
            output
                << ','
                << to_csv(experiment.config.block_orthogonalization) << ','
                << to_csv(experiment.config.partial_cholesky_stopping_rule) << ','
                << experiment.config.partial_cholesky_condition_limit << ','
                << (experiment.config.adaptive_s ? "true" : "false") << ','
                << experiment.config.s_min << ','
                << experiment.config.s_max << ','
                << experiment.config.s_grow_after << ','
                << (experiment.config.s_initial_probe ? "true" : "false") << ','
                << to_csv(experiment.config.polynomial_basis) << ','
                << (experiment.config.enable_recycling ? "true" : "false") << ','
                << experiment.config.recycle_count << ','
                << (experiment.config.verbose ? "true" : "false") << ','
                << std::quoted(experiment.initial_guess_description) << ','
                << (experiment.has_exact_solution ? "true" : "false") << ','
                << std::quoted(experiment.exact_solution_description) << ','
                << std::quoted(experiment.right_hand_side_description) << ','
                << (experiment.write_solution_output ? "true" : "false") << '\n';
        }

        /**
         * @brief Writes the experiment convergence-history CSV file.
         * @param path Destination CSV path.
         * @param experiment Experiment metadata and residual history.
         */
        void write_experiment_convergence(
            const std::filesystem::path& path,
            const GMRESExperimentData& experiment)
        {
            std::ofstream output = open_csv(path);

            write_common_header(output);
            output
                << ",converged,total_iterations,blocks_completed,"
                   "history_entry,iteration,block_s,residual_kind,"
                   "residual_norm,residual_relative_to_initial\n";

            for (const ResidualHistoryEntry& entry :
                 experiment.residual_history)
            {
                const Scalar relative_residual =
                    experiment.initial_residual == 0.0
                        ? 0.0
                        : entry.residual_norm / experiment.initial_residual;

                write_common_values(output, experiment);
                output
                    << ','
                    << (experiment.converged ? "true" : "false") << ','
                    << experiment.iterations << ','
                    << experiment.blocks_completed << ','
                    << entry.history_entry << ','
                    << entry.iteration << ','
                    << entry.block_s << ','
                    << entry.kind << ','
                    << entry.residual_norm << ','
                    << relative_residual << '\n';
            }
        }

        /**
         * @brief Writes the experiment performance CSV file.
         * @param path Destination CSV path.
         * @param experiment Experiment metadata and timing data.
         */
        void write_experiment_performance(
            const std::filesystem::path& path,
            const GMRESExperimentData& experiment)
        {
            std::ofstream output = open_csv(path);

            write_common_header(output);
            output
                << ",converged,iterations,blocks_completed,"
                    "residual_history_entries,solve_seconds,"
                    "seconds_per_iteration,halo_statistics_available,"
                    "halo_recv_peers_mean,halo_recv_peers_max,"
                    "halo_send_peers_mean,halo_send_peers_max,"
                    "halo_recv_values_mean,halo_recv_values_max,"
                    "halo_send_values_mean,halo_send_values_max\n";

            write_common_values(output, experiment);
            output
                << ','
                << (experiment.converged ? "true" : "false") << ','
                << experiment.iterations << ','
                << experiment.blocks_completed << ','
                << experiment.residual_history.size() << ','
                << experiment.solve_seconds << ',';

            if (experiment.iterations != 0)
            {
                output
                    << experiment.solve_seconds
                        / static_cast<double>(experiment.iterations);
            }

            output << ','
                   << (experiment.halo_exchange.available ? "true" : "false");

            if (experiment.halo_exchange.available)
            {
                output
                    << ',' << experiment.halo_exchange.mean_receive_peers
                    << ',' << experiment.halo_exchange.max_receive_peers
                    << ',' << experiment.halo_exchange.mean_send_peers
                    << ',' << experiment.halo_exchange.max_send_peers
                    << ',' << experiment.halo_exchange.mean_receive_values
                    << ',' << experiment.halo_exchange.max_receive_values
                    << ',' << experiment.halo_exchange.mean_send_values
                    << ',' << experiment.halo_exchange.max_send_values;
            }
            else
            {
                output << ",,,,,,,,";
            }

            output << '\n';
        }

        /**
         * @brief Writes the experiment accuracy CSV file.
         * @param path Destination CSV path.
         * @param experiment Experiment metadata and accuracy data.
         */
        void write_experiment_accuracy(
            const std::filesystem::path& path,
            const GMRESExperimentData& experiment)
        {
            std::ofstream output = open_csv(path);

            write_common_header(output);
            output
                << ",converged,iterations,blocks_completed,"
                   "initial_residual,final_residual,residual_reduction,"
                   "relative_residual,relative_forward_error\n";

            write_common_values(output, experiment);
            output
                << ','
                << (experiment.converged ? "true" : "false") << ','
                << experiment.iterations << ','
                << experiment.blocks_completed << ','
                << experiment.initial_residual << ','
                << experiment.final_residual << ',';

            if (experiment.initial_residual != 0.0)
            {
                output
                    << experiment.final_residual
                        / experiment.initial_residual;
            }

            output
                << ','
                << experiment.relative_residual << ',';

            if (experiment.has_exact_solution)
            {
                output << experiment.relative_forward_error;
            }

            output << '\n';
        }

        /**
         * @brief Writes the final-solution CSV file.
         * @param path Destination CSV path.
         * @param experiment Experiment metadata and solution vectors.
         */
        void write_experiment_solution(
            const std::filesystem::path& path,
            const GMRESExperimentData& experiment)
        {
            if (experiment.has_exact_solution
                && experiment.solution.size()
                    != experiment.exact_solution.size())
            {
                throw std::invalid_argument(
                    "Computed and exact solutions must have the same size.");
            }

            std::ofstream output = open_csv(path);

            write_common_header(output);
            output
                << ",converged,iterations,blocks_completed,index,"
                   "computed_value,exact_value,signed_error,"
                   "absolute_error\n";

            for (Index i = 0; i < experiment.solution.size(); ++i)
            {
                write_common_values(output, experiment);
                output
                    << ','
                    << (experiment.converged ? "true" : "false") << ','
                    << experiment.iterations << ','
                    << experiment.blocks_completed << ','
                    << i << ','
                    << experiment.solution[i];

                if (experiment.has_exact_solution)
                {
                    const Scalar signed_error =
                        experiment.solution[i] - experiment.exact_solution[i];

                    output
                        << ',' << experiment.exact_solution[i]
                        << ',' << signed_error
                        << ',' << std::abs(signed_error);
                }
                else
                {
                    output << ",,,";
                }

                output << '\n';
            }
        }
    }

    SparseMatrixCSR read_matrix_market(const std::string& filename)
    {
        std::ifstream file(filename);

        if (!file)
        {
            throw std::runtime_error("Could not open Matrix Market file: " + filename);
        }

        std::string header;

        if (!std::getline(file, header))
        {
            throw std::runtime_error("Matrix Market file is empty: " + filename);
        }

        const std::string normalized_header = lowercase(header);

        if (normalized_header.find("%%matrixmarket") != 0
            || normalized_header.find("matrix") == std::string::npos
            || normalized_header.find("coordinate") == std::string::npos)
        {
            throw std::runtime_error(
                "Only Matrix Market coordinate matrices are supported: "
                + filename);
        }

        const bool symmetric =
            normalized_header.find("symmetric") != std::string::npos;
        const bool pattern =
            normalized_header.find("pattern") != std::string::npos;
        const std::string size_text =
            read_matrix_market_data_line(file, filename);

        Index rows = 0;
        Index cols = 0;
        Index nonzeros = 0;

        {
            std::stringstream size_line(size_text);
            size_line >> rows >> cols >> nonzeros;

            if (!size_line)
            {
                throw std::runtime_error("Invalid matrix size line in file: " + filename);
            }
        }

        std::vector<std::tuple<Index, Index, Scalar>> entries;
        entries.reserve(symmetric ? 2 * nonzeros : nonzeros);

        for (Index k = 0; k < nonzeros; ++k)
        {
            Index row = 0;
            Index col = 0;
            Scalar value = 1.0;

            file >> row >> col;

            if (!pattern)
            {
                file >> value;
            }

            if (!file)
            {
                throw std::runtime_error("Invalid matrix entry in file: " + filename);
            }

            if (row == 0 || col == 0 || row > rows || col > cols)
            {
                throw std::runtime_error(
                    "Matrix Market matrix index out of range: " + filename);
            }

            const Index zero_based_row = row - 1;
            const Index zero_based_col = col - 1;

            entries.emplace_back(
                zero_based_row,
                zero_based_col,
                value);

            if (symmetric && zero_based_row != zero_based_col)
            {
                entries.emplace_back(
                    zero_based_col,
                    zero_based_row,
                    value);
            }
        }

        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b)
                  {
                      if (std::get<0>(a) == std::get<0>(b))
                      {
                          return std::get<1>(a) < std::get<1>(b);
                      }

                      return std::get<0>(a) < std::get<0>(b);
                  });

        Vector values;
        std::vector<Index> col_indices;
        std::vector<Index> row_ptr(rows + 1, 0);

        values.reserve(entries.size());
        col_indices.reserve(entries.size());

        for (const auto& entry : entries)
        {
            Index row = std::get<0>(entry);
            Index col = std::get<1>(entry);
            Scalar value = std::get<2>(entry);

            values.push_back(value);
            col_indices.push_back(col);
            row_ptr[row + 1]++;
        }

        for (Index i = 0; i < rows; ++i)
        {
            row_ptr[i + 1] += row_ptr[i];
        }

        return SparseMatrixCSR(rows, cols, values, col_indices, row_ptr);
    }

    Vector read_matrix_market_vector(const std::string& filename)
    {
        std::ifstream file(filename);

        if (!file)
        {
            throw std::runtime_error(
                "Could not open Matrix Market vector file: " + filename);
        }

        std::string header;

        if (!std::getline(file, header))
        {
            throw std::runtime_error(
                "Matrix Market vector file is empty: " + filename);
        }

        const std::string normalized_header = lowercase(header);

        if (normalized_header.find("%%matrixmarket") != 0
            || normalized_header.find("matrix") == std::string::npos)
        {
            throw std::runtime_error(
                "Invalid Matrix Market vector header: " + filename);
        }

        const bool coordinate =
            normalized_header.find("coordinate") != std::string::npos;
        const bool array =
            normalized_header.find("array") != std::string::npos;
        const bool pattern =
            normalized_header.find("pattern") != std::string::npos;

        const std::string size_text =
            read_matrix_market_data_line(file, filename);
        std::stringstream size_line(size_text);

        Index rows = 0;
        Index cols = 0;

        if (!(size_line >> rows >> cols))
        {
            throw std::runtime_error(
                "Invalid Matrix Market vector size line: " + filename);
        }

        if (rows == 0 || cols == 0)
        {
            throw std::runtime_error(
                "Matrix Market vector has a zero dimension: " + filename);
        }

        // Use the first column of a multi-column Matrix Market array.
        const bool column_vector = rows > 1;
        const Index vector_size = column_vector ? rows : cols;
        Vector values(vector_size, 0.0);

        if (coordinate)
        {
            Index nonzeros = 0;

            if (!(size_line >> nonzeros))
            {
                throw std::runtime_error(
                    "Invalid Matrix Market coordinate vector size line: "
                    + filename);
            }

            for (Index entry = 0; entry < nonzeros; ++entry)
            {
                Index row = 0;
                Index col = 0;
                Scalar value = 1.0;

                file >> row >> col;

                if (!pattern)
                {
                    file >> value;
                }

                if (!file)
                {
                    throw std::runtime_error(
                        "Invalid Matrix Market vector entry in file: "
                        + filename);
                }

                if (row == 0 || col == 0 || row > rows || col > cols)
                {
                    throw std::runtime_error(
                        "Matrix Market vector index out of range: "
                        + filename);
                }

                // Ignore additional right-hand-side columns.
                if (column_vector)
                {
                    if (col == 1)
                    {
                        values[row - 1] = value;
                    }
                }
                else
                {
                    values[col - 1] = value;
                }
            }

            return values;
        }

        if (array)
        {
            // Matrix Market arrays store the first column first.
            for (Scalar& value : values)
            {
                file >> value;

                if (!file)
                {
                    throw std::runtime_error(
                        "Invalid Matrix Market array vector entry in file: "
                        + filename);
                }
            }

            return values;
        }

        throw std::runtime_error(
            "Unsupported Matrix Market vector storage format: " + filename);
    }

    void write_residual_history(const std::string& filename,
                                const Vector& residuals)
    {
        std::ofstream file(filename);

        if (!file)
        {
            throw std::runtime_error("Could not open output file: " + filename);
        }

        file << "iteration,residual\n";

        for (Index i = 0; i < residuals.size(); ++i)
        {
            file << i << "," << residuals[i] << "\n";
        }
    }

    void write_vector_csv(const std::string& filename,
                          const Vector& x)
    {
        std::ofstream file(filename);

        if (!file)
        {
            throw std::runtime_error("Could not open output file: " + filename);
        }

        file << "index,value\n";

        for (Index i = 0; i < x.size(); ++i)
        {
            file << i << "," << x[i] << "\n";
        }
    }

    std::vector<ResidualHistoryEntry> make_restarted_residual_history(
        const Vector& residuals,
        Index iterations,
        Index restart)
    {
        if (restart == 0)
        {
            throw std::invalid_argument(
                "Restart must be greater than zero.");
        }

        std::vector<ResidualHistoryEntry> history;
        history.reserve(residuals.size());

        if (residuals.empty())
        {
            return history;
        }

        Index history_entry = 0;
        Index iteration = 0;

        history.push_back({
            history_entry,
            iteration,
            "initial",
            residuals[history_entry]
        });
        ++history_entry;

        while (history_entry < residuals.size()
               && iteration < iterations)
        {
            const Index cycle_iterations =
                std::min(restart, iterations - iteration);

            for (Index inner_iteration = 0;
                 inner_iteration < cycle_iterations
                 && history_entry < residuals.size();
                 ++inner_iteration)
            {
                ++iteration;
                history.push_back({
                    history_entry,
                    iteration,
                    "estimated",
                    residuals[history_entry]
                });
                ++history_entry;
            }

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

    void write_gmres_experiment_outputs(
        const std::filesystem::path& output_directory,
        const GMRESExperimentData& experiment)
    {
        if (experiment.output_prefix.empty())
        {
            throw std::invalid_argument(
                "Experiment output prefix must not be empty.");
        }

        if (experiment.process_count == 0)
        {
            throw std::invalid_argument(
                "Experiment process count must be greater than zero.");
        }

        const std::string matrix_name = make_filename_component(
            std::filesystem::path(experiment.matrix_path).stem().string());
        const std::string output_prefix = make_filename_component(
            experiment.output_prefix);
        const std::string filename_prefix =
            matrix_name
            + "_"
            + output_prefix
            + "_p"
            + std::to_string(experiment.process_count);

        const std::filesystem::path config_directory =
            output_directory / "config";
        const std::filesystem::path convergence_directory =
            output_directory / "convergence";
        const std::filesystem::path performance_directory =
            output_directory / "performance";
        const std::filesystem::path accuracy_directory =
            output_directory / "accuracy";

        std::filesystem::create_directories(config_directory);
        std::filesystem::create_directories(convergence_directory);
        std::filesystem::create_directories(performance_directory);
        std::filesystem::create_directories(accuracy_directory);
        if (experiment.write_solution_output)
        {
            std::filesystem::create_directories(output_directory / "solution");
        }

        write_experiment_config(
            config_directory
                / (filename_prefix + "_config.csv"),
            experiment);
        write_experiment_convergence(
            convergence_directory
                / (filename_prefix + "_convergence.csv"),
            experiment);
        write_experiment_performance(
            performance_directory
                / (filename_prefix + "_performance.csv"),
            experiment);
        write_experiment_accuracy(
            accuracy_directory
                / (filename_prefix + "_accuracy.csv"),
            experiment);
        if (experiment.write_solution_output)
        {
            write_experiment_solution(
                output_directory / "solution"
                    / (filename_prefix + "_solution.csv"),
                experiment);
        }
    }
}
