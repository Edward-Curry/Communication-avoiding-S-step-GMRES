#include "parallel/distributed_io.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
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
        using WireIndex = unsigned long long;

        struct MatrixEntry
        {
            Index row = 0;
            Index col = 0;
            Scalar value = 0.0;
        };

        struct MatrixMarketMatrix
        {
            Index rows = 0;
            Index cols = 0;
            std::vector<MatrixEntry> entries;
        };

        std::string lower_copy(std::string text)
        {
            std::transform(text.begin(),
                           text.end(),
                           text.begin(),
                           [](unsigned char c)
                           {
                               return static_cast<char>(std::tolower(c));
                           });

            return text;
        }

        std::string read_data_line(std::ifstream& file,
                                   const std::string& filename)
        {
            std::string line;

            do
            {
                if (!std::getline(file, line))
                {
                    throw std::runtime_error("Missing Matrix Market size line in file: " + filename);
                }
            } while (line.empty() || line[0] == '%');

            return line;
        }

        Index local_size_for_rank(Index global_size,
                                  int rank,
                                  int comm_size)
        {
            const Index base = global_size / static_cast<Index>(comm_size);
            const Index remainder = global_size % static_cast<Index>(comm_size);

            return base + (static_cast<Index>(rank) < remainder ? 1 : 0);
        }

        Index local_start_for_rank(Index global_size,
                                   int rank,
                                   int comm_size)
        {
            const Index base = global_size / static_cast<Index>(comm_size);
            const Index remainder = global_size % static_cast<Index>(comm_size);
            const Index rank_index = static_cast<Index>(rank);

            return rank_index * base + std::min(rank_index, remainder);
        }

        int owner_rank_for_row(Index row,
                               Index global_rows,
                               int comm_size)
        {
            const Index base = global_rows / static_cast<Index>(comm_size);
            const Index remainder = global_rows % static_cast<Index>(comm_size);
            const Index longer_rows = (base + 1) * remainder;

            if (row < longer_rows)
            {
                return static_cast<int>(row / (base + 1));
            }

            if (base == 0)
            {
                throw std::runtime_error("Could not assign row to MPI rank.");
            }

            return static_cast<int>(remainder + (row - longer_rows) / base);
        }

        int checked_int(Index value,
                        const std::string& description)
        {
            if (value > static_cast<Index>(std::numeric_limits<int>::max()))
            {
                throw std::runtime_error(description + " exceeds MPI int range.");
            }

            return static_cast<int>(value);
        }

        MatrixMarketMatrix read_matrix_market_on_root(const std::string& filename)
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

            const std::string lower_header = lower_copy(header);

            if (lower_header.find("%%matrixmarket") != 0 ||
                lower_header.find("matrix") == std::string::npos ||
                lower_header.find("coordinate") == std::string::npos)
            {
                throw std::runtime_error("Only Matrix Market coordinate matrices are supported: " + filename);
            }

            const bool symmetric =
                lower_header.find("symmetric") != std::string::npos;
            const bool pattern =
                lower_header.find("pattern") != std::string::npos;

            const std::string size_line_text = read_data_line(file, filename);
            std::stringstream size_line(size_line_text);

            MatrixMarketMatrix matrix;
            Index nonzeros = 0;

            if (!(size_line >> matrix.rows >> matrix.cols >> nonzeros))
            {
                throw std::runtime_error("Invalid Matrix Market matrix size line: " + filename);
            }

            matrix.entries.reserve(symmetric ? 2 * nonzeros : nonzeros);

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
                    throw std::runtime_error("Invalid Matrix Market matrix entry in file: " + filename);
                }

                if (row == 0 || col == 0 || row > matrix.rows || col > matrix.cols)
                {
                    throw std::runtime_error("Matrix Market matrix index out of range: " + filename);
                }

                const Index zero_based_row = row - 1;
                const Index zero_based_col = col - 1;

                matrix.entries.push_back({zero_based_row, zero_based_col, value});

                if (symmetric && zero_based_row != zero_based_col)
                {
                    matrix.entries.push_back({zero_based_col, zero_based_row, value});
                }
            }

            return matrix;
        }

        Vector read_matrix_market_vector_on_root(const std::string& filename)
        {
            std::ifstream file(filename);

            if (!file)
            {
                throw std::runtime_error("Could not open Matrix Market vector file: " + filename);
            }

            std::string header;

            if (!std::getline(file, header))
            {
                throw std::runtime_error("Matrix Market vector file is empty: " + filename);
            }

            const std::string lower_header = lower_copy(header);

            if (lower_header.find("%%matrixmarket") != 0 ||
                lower_header.find("matrix") == std::string::npos)
            {
                throw std::runtime_error("Invalid Matrix Market vector header: " + filename);
            }

            const bool coordinate =
                lower_header.find("coordinate") != std::string::npos;
            const bool array =
                lower_header.find("array") != std::string::npos;
            const bool pattern =
                lower_header.find("pattern") != std::string::npos;

            const std::string size_line_text = read_data_line(file, filename);
            std::stringstream size_line(size_line_text);

            Index rows = 0;
            Index cols = 0;

            if (!(size_line >> rows >> cols))
            {
                throw std::runtime_error("Invalid Matrix Market vector size line: " + filename);
            }

            if (rows != 1 && cols != 1)
            {
                throw std::runtime_error("Matrix Market vector must have one row or one column: " + filename);
            }

            const bool column_vector = cols == 1;
            const Index vector_size = column_vector ? rows : cols;
            Vector values(vector_size, 0.0);

            if (coordinate)
            {
                Index nonzeros = 0;

                if (!(size_line >> nonzeros))
                {
                    throw std::runtime_error("Invalid Matrix Market coordinate vector size line: " + filename);
                }

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
                        throw std::runtime_error("Invalid Matrix Market vector entry in file: " + filename);
                    }

                    if (row == 0 || col == 0 || row > rows || col > cols)
                    {
                        throw std::runtime_error("Matrix Market vector index out of range: " + filename);
                    }

                    const Index index = column_vector ? row - 1 : col - 1;
                    values[index] = value;
                }

                return values;
            }

            if (array)
            {
                for (Index i = 0; i < vector_size; ++i)
                {
                    file >> values[i];

                    if (!file)
                    {
                        throw std::runtime_error("Invalid Matrix Market array vector entry in file: " + filename);
                    }
                }

                return values;
            }

            throw std::runtime_error("Unsupported Matrix Market vector storage format: " + filename);
        }

        void broadcast_read_status(int failed,
                                   const std::string& message,
                                   MPI_Comm comm)
        {
            int rank = 0;
            MPI_Comm_rank(comm, &rank);

            MPI_Bcast(&failed, 1, MPI_INT, 0, comm);

            int message_size = rank == 0
                                   ? static_cast<int>(message.size())
                                   : 0;

            MPI_Bcast(&message_size, 1, MPI_INT, 0, comm);

            std::vector<char> buffer(message_size);

            if (rank == 0 && message_size > 0)
            {
                std::copy(message.begin(), message.end(), buffer.begin());
            }

            if (message_size > 0)
            {
                MPI_Bcast(buffer.data(), message_size, MPI_CHAR, 0, comm);
            }

            if (failed)
            {
                throw std::runtime_error(std::string(buffer.begin(), buffer.end()));
            }
        }
    }

    DistributedSparseMatrixCSR read_matrix_market_distributed(const std::string& filename,
                                                              MPI_Comm comm)
    {
        if (comm == MPI_COMM_NULL)
        {
            throw std::invalid_argument("read_matrix_market_distributed: communicator cannot be MPI_COMM_NULL.");
        }

        int rank = 0;
        int comm_size = 1;

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &comm_size);

        WireIndex global_rows_wire = 0;
        WireIndex global_cols_wire = 0;

        std::vector<int> send_counts;
        std::vector<int> displacements;
        std::vector<WireIndex> send_rows;
        std::vector<WireIndex> send_cols;
        Vector send_values;

        int failed = 0;
        std::string error_message;

        if (rank == 0)
        {
            try
            {
                MatrixMarketMatrix matrix = read_matrix_market_on_root(filename);

                global_rows_wire = static_cast<WireIndex>(matrix.rows);
                global_cols_wire = static_cast<WireIndex>(matrix.cols);

                send_counts.assign(comm_size, 0);

                for (const MatrixEntry& entry : matrix.entries)
                {
                    const int owner =
                        owner_rank_for_row(entry.row, matrix.rows, comm_size);
                    ++send_counts[owner];
                }

                displacements.assign(comm_size, 0);

                for (int r = 1; r < comm_size; ++r)
                {
                    displacements[r] = displacements[r - 1] + send_counts[r - 1];
                }

                const Index total_entries = matrix.entries.size();

                checked_int(total_entries, "Number of scattered matrix entries");

                send_rows.assign(total_entries, 0);
                send_cols.assign(total_entries, 0);
                send_values.assign(total_entries, 0.0);

                std::vector<int> offsets = displacements;

                for (const MatrixEntry& entry : matrix.entries)
                {
                    const int owner =
                        owner_rank_for_row(entry.row, matrix.rows, comm_size);
                    const Index owner_start =
                        local_start_for_rank(matrix.rows, owner, comm_size);
                    const int position = offsets[owner]++;

                    send_rows[position] =
                        static_cast<WireIndex>(entry.row - owner_start);
                    send_cols[position] = static_cast<WireIndex>(entry.col);
                    send_values[position] = entry.value;
                }
            }
            catch (const std::exception& error)
            {
                failed = 1;
                error_message = error.what();
            }
        }

        broadcast_read_status(failed, error_message, comm);

        MPI_Bcast(&global_rows_wire, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);
        MPI_Bcast(&global_cols_wire, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);

        const Index global_rows = static_cast<Index>(global_rows_wire);
        const Index global_cols = static_cast<Index>(global_cols_wire);
        const Index local_row_start =
            local_start_for_rank(global_rows, rank, comm_size);
        const Index local_rows =
            local_size_for_rank(global_rows, rank, comm_size);

        int local_nnz = 0;

        MPI_Scatter(rank == 0 ? send_counts.data() : nullptr,
                    1,
                    MPI_INT,
                    &local_nnz,
                    1,
                    MPI_INT,
                    0,
                    comm);

        std::vector<WireIndex> local_row_indices(local_nnz, 0);
        std::vector<WireIndex> local_col_indices(local_nnz, 0);
        Vector local_values(local_nnz, 0.0);

        MPI_Scatterv(rank == 0 ? send_rows.data() : nullptr,
                     rank == 0 ? send_counts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr,
                     MPI_UNSIGNED_LONG_LONG,
                     local_row_indices.data(),
                     local_nnz,
                     MPI_UNSIGNED_LONG_LONG,
                     0,
                     comm);

        MPI_Scatterv(rank == 0 ? send_cols.data() : nullptr,
                     rank == 0 ? send_counts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr,
                     MPI_UNSIGNED_LONG_LONG,
                     local_col_indices.data(),
                     local_nnz,
                     MPI_UNSIGNED_LONG_LONG,
                     0,
                     comm);

        MPI_Scatterv(rank == 0 ? send_values.data() : nullptr,
                     rank == 0 ? send_counts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr,
                     MPI_DOUBLE,
                     local_values.data(),
                     local_nnz,
                     MPI_DOUBLE,
                     0,
                     comm);

        std::vector<std::tuple<Index, Index, Scalar>> entries;
        entries.reserve(static_cast<Index>(local_nnz));

        for (int i = 0; i < local_nnz; ++i)
        {
            entries.emplace_back(static_cast<Index>(local_row_indices[i]),
                                 static_cast<Index>(local_col_indices[i]),
                                 local_values[i]);
        }

        std::sort(entries.begin(),
                  entries.end(),
                  [](const auto& left, const auto& right)
                  {
                      if (std::get<0>(left) == std::get<0>(right))
                      {
                          return std::get<1>(left) < std::get<1>(right);
                      }

                      return std::get<0>(left) < std::get<0>(right);
                  });

        Vector values;
        std::vector<Index> col_indices;
        std::vector<Index> row_ptr(local_rows + 1, 0);

        values.reserve(entries.size());
        col_indices.reserve(entries.size());

        for (const auto& entry : entries)
        {
            const Index local_row = std::get<0>(entry);
            const Index col = std::get<1>(entry);
            const Scalar value = std::get<2>(entry);

            values.push_back(value);
            col_indices.push_back(col);
            ++row_ptr[local_row + 1];
        }

        for (Index i = 0; i < local_rows; ++i)
        {
            row_ptr[i + 1] += row_ptr[i];
        }

        return DistributedSparseMatrixCSR(global_rows,
                                          global_cols,
                                          local_row_start,
                                          local_rows,
                                          values,
                                          col_indices,
                                          row_ptr,
                                          comm);
    }

    DistributedVector read_matrix_market_vector_distributed(const std::string& filename,
                                                            MPI_Comm comm)
    {
        if (comm == MPI_COMM_NULL)
        {
            throw std::invalid_argument("read_matrix_market_vector_distributed: communicator cannot be MPI_COMM_NULL.");
        }

        int rank = 0;
        int comm_size = 1;

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &comm_size);

        Vector global_values;
        WireIndex global_size_wire = 0;
        int failed = 0;
        std::string error_message;

        if (rank == 0)
        {
            try
            {
                global_values = read_matrix_market_vector_on_root(filename);
                global_size_wire =
                    static_cast<WireIndex>(global_values.size());
            }
            catch (const std::exception& error)
            {
                failed = 1;
                error_message = error.what();
            }
        }

        broadcast_read_status(failed, error_message, comm);

        MPI_Bcast(&global_size_wire, 1, MPI_UNSIGNED_LONG_LONG, 0, comm);

        const Index global_size = static_cast<Index>(global_size_wire);
        const Index local_start =
            local_start_for_rank(global_size, rank, comm_size);
        const Index local_rows =
            local_size_for_rank(global_size, rank, comm_size);

        std::vector<int> send_counts;
        std::vector<int> displacements;

        if (rank == 0)
        {
            send_counts.assign(comm_size, 0);
            displacements.assign(comm_size, 0);

            for (int r = 0; r < comm_size; ++r)
            {
                send_counts[r] =
                    checked_int(local_size_for_rank(global_size, r, comm_size),
                                "Distributed vector local size");
            }

            for (int r = 1; r < comm_size; ++r)
            {
                displacements[r] = displacements[r - 1] + send_counts[r - 1];
            }
        }

        Vector local_values(local_rows, 0.0);

        MPI_Scatterv(rank == 0 ? global_values.data() : nullptr,
                     rank == 0 ? send_counts.data() : nullptr,
                     rank == 0 ? displacements.data() : nullptr,
                     MPI_DOUBLE,
                     local_values.data(),
                     checked_int(local_rows, "Distributed vector local size"),
                     MPI_DOUBLE,
                     0,
                     comm);

        return DistributedVector(global_size,
                                 local_start,
                                 local_values,
                                 comm);
    }

    DistributedVector make_distributed_vector(const Vector& global_values,
                                              Index local_start,
                                              Index local_rows,
                                              MPI_Comm comm)
    {
        if (local_start + local_rows > global_values.size())
        {
            throw std::invalid_argument("make_distributed_vector: local range exceeds global vector size.");
        }

        Vector local_values(local_rows, 0.0);

        for (Index i = 0; i < local_rows; ++i)
        {
            local_values[i] = global_values[local_start + i];
        }

        return DistributedVector(global_values.size(),
                                 local_start,
                                 local_values,
                                 comm);
    }

    Vector gather_distributed_vector(const DistributedVector& x,
                                     MPI_Comm comm,
                                     int root)
    {
        if (comm == MPI_COMM_NULL)
        {
            throw std::invalid_argument("gather_distributed_vector: communicator cannot be MPI_COMM_NULL.");
        }

        int rank = 0;
        int comm_size = 1;

        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &comm_size);

        const int local_count =
            checked_int(x.local_size(), "Distributed vector local size");
        std::vector<int> recv_counts(comm_size, 0);

        MPI_Gather(&local_count,
                   1,
                   MPI_INT,
                   recv_counts.data(),
                   1,
                   MPI_INT,
                   root,
                   comm);

        std::vector<int> displacements(comm_size, 0);

        if (rank == root)
        {
            for (int r = 1; r < comm_size; ++r)
            {
                displacements[r] = displacements[r - 1] + recv_counts[r - 1];
            }
        }

        Vector global_values;

        if (rank == root)
        {
            global_values.assign(x.global_size(), 0.0);
        }

        MPI_Gatherv(x.local_values().data(),
                    local_count,
                    MPI_DOUBLE,
                    rank == root ? global_values.data() : nullptr,
                    recv_counts.data(),
                    displacements.data(),
                    MPI_DOUBLE,
                    root,
                    comm);

        return global_values;
    }
}
