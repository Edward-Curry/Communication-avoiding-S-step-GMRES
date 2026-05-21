#include "common/io.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace gmres
{
    SparseMatrixCSR read_matrix_market(const std::string& filename)
    {
        std::ifstream file(filename);

        if (!file)
        {
            throw std::runtime_error("Could not open Matrix Market file: " + filename);
        }

        std::string line;

        // Read header line
        if (!std::getline(file, line))
        {
            throw std::runtime_error("Matrix Market file is empty: " + filename);
        }

        if (line.find("%%MatrixMarket") != 0)
        {
            throw std::runtime_error("Invalid Matrix Market header in file: " + filename);
        }

        // Skip comments
        do
        {
            if (!std::getline(file, line))
            {
                throw std::runtime_error("Missing matrix size line in file: " + filename);
            }
        } while (!line.empty() && line[0] == '%');

        Index rows = 0;
        Index cols = 0;
        Index nonzeros = 0;

        {
            std::stringstream size_line(line);
            size_line >> rows >> cols >> nonzeros;

            if (!size_line)
            {
                throw std::runtime_error("Invalid matrix size line in file: " + filename);
            }
        }

        std::vector<std::tuple<Index, Index, Scalar>> entries;
        entries.reserve(nonzeros);

        for (Index k = 0; k < nonzeros; ++k)
        {
            Index row = 0;
            Index col = 0;
            Scalar value = 0.0;

            file >> row >> col >> value;

            if (!file)
            {
                throw std::runtime_error("Invalid matrix entry in file: " + filename);
            }

            // Matrix Market uses 1-based indexing.
            // C++ uses 0-based indexing.
            entries.emplace_back(row - 1, col - 1, value);
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

        values.reserve(nonzeros);
        col_indices.reserve(nonzeros);

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
}