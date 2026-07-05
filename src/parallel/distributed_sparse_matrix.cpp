#include "parallel/distributed_sparse_matrix.hpp"

#include <algorithm>
#include <limits>
#include <string>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace gmres
{
    namespace
    {
        using WireIndex = unsigned long long;

        constexpr int halo_exchange_tag = 27183;

        int checked_int(Index value, const char* description)
        {
            if (value > static_cast<Index>(std::numeric_limits<int>::max()))
            {
                throw std::length_error(std::string(description) +
                                        " exceeds the MPI count range.");
            }

            return static_cast<int>(value);
        }

        WireIndex to_wire_index(Index value)
        {
            return static_cast<WireIndex>(value);
        }
    }

    DistributedSparseMatrixCSR::DistributedSparseMatrixCSR()
        : global_rows_(0),
          global_cols_(0),
          local_row_start_(0),
          local_rows_(0),
          values_(),
          col_indices_(),
          row_ptr_(),
          comm_(MPI_COMM_WORLD)
    {
    }

    DistributedSparseMatrixCSR::DistributedSparseMatrixCSR(
        Index global_rows,
        Index global_cols,
        Index local_row_start,
        Index local_rows,
        const Vector& values,
        const std::vector<Index>& col_indices,
        const std::vector<Index>& row_ptr,
        MPI_Comm comm)
        : global_rows_(global_rows),
          global_cols_(global_cols),
          local_row_start_(local_row_start),
          local_rows_(local_rows),
          values_(values),
          col_indices_(col_indices),
          row_ptr_(row_ptr),
          comm_(comm)
    {
        if (comm_ == MPI_COMM_NULL)
        {
            throw std::invalid_argument("DistributedSparseMatrixCSR communicator cannot be MPI_COMM_NULL.");
        }

        if (local_row_start_ > global_rows_)
        {
            throw std::invalid_argument("Local row start exceeds global row count.");
        }

        if (local_row_start_ + local_rows_ > global_rows_)
        {
            throw std::invalid_argument("Local row range exceeds global row count.");
        }

        if (row_ptr_.size() != local_rows_ + 1)
        {
            throw std::invalid_argument("row_ptr size must be local_rows + 1.");
        }

        if (values_.size() != col_indices_.size())
        {
            throw std::invalid_argument("values and col_indices must have the same size.");
        }

        if (!row_ptr_.empty() && row_ptr_.back() != values_.size())
        {
            throw std::invalid_argument("Last row_ptr entry must equal number of local nonzeros.");
        }

        for (Index col : col_indices_)
        {
            if (col >= global_cols_)
            {
                throw std::invalid_argument("Column index exceeds global column count.");
            }
        }
    }

    Index DistributedSparseMatrixCSR::global_rows() const
    {
        return global_rows_;
    }

    Index DistributedSparseMatrixCSR::global_cols() const
    {
        return global_cols_;
    }

    Index DistributedSparseMatrixCSR::local_row_start() const
    {
        return local_row_start_;
    }

    Index DistributedSparseMatrixCSR::local_rows() const
    {
        return local_rows_;
    }

    Index DistributedSparseMatrixCSR::local_row_end() const
    {
        return local_row_start_ + local_rows_;
    }

    Index DistributedSparseMatrixCSR::nonzeros() const
    {
        return values_.size();
    }

    MPI_Comm DistributedSparseMatrixCSR::communicator() const
    {
        return comm_;
    }

    void DistributedSparseMatrixCSR::initialize_halo_plan(
        const DistributedVector& x) const
    {
        int comm_size = 0;
        MPI_Comm_size(comm_, &comm_size);

        const WireIndex local_layout[2] = {
            to_wire_index(x.local_start()),
            to_wire_index(x.local_size())
        };

        std::vector<WireIndex> layouts(static_cast<Index>(comm_size) * 2, 0);
        MPI_Allgather(local_layout,
                      2,
                      MPI_UNSIGNED_LONG_LONG,
                      layouts.data(),
                      2,
                      MPI_UNSIGNED_LONG_LONG,
                      comm_);

        std::vector<Index> starts(comm_size, 0);
        std::vector<Index> sizes(comm_size, 0);
        Index expected_start = 0;

        for (int rank = 0; rank < comm_size; ++rank)
        {
            starts[rank] = static_cast<Index>(layouts[2 * rank]);
            sizes[rank] = static_cast<Index>(layouts[2 * rank + 1]);

            if (starts[rank] != expected_start)
            {
                throw std::invalid_argument(
                    "Distributed vector ranges must form a contiguous partition.");
            }

            expected_start += sizes[rank];
        }

        if (expected_start != x.global_size())
        {
            throw std::invalid_argument(
                "Distributed vector ranges do not cover the global vector.");
        }

        auto owner_of = [&](Index global_index)
        {
            auto upper = std::upper_bound(starts.begin(),
                                          starts.end(),
                                          global_index);

            if (upper == starts.begin())
            {
                throw std::runtime_error("Unable to find owner of vector entry.");
            }

            const int owner =
                static_cast<int>(upper - starts.begin() - 1);

            if (global_index >= starts[owner] + sizes[owner])
            {
                throw std::runtime_error("Unable to find owner of vector entry.");
            }

            return owner;
        };

        std::vector<std::vector<Index>> requested_indices(comm_size);

        for (Index column : col_indices_)
        {
            if (column >= x.local_start() && column < x.local_end())
            {
                continue;
            }

            requested_indices[owner_of(column)].push_back(column);
        }

        halo_receive_counts_.assign(comm_size, 0);
        halo_receive_displacements_.assign(comm_size, 0);
        Index total_receive_count = 0;

        for (int rank = 0; rank < comm_size; ++rank)
        {
            auto& indices = requested_indices[rank];
            std::sort(indices.begin(), indices.end());
            indices.erase(std::unique(indices.begin(), indices.end()),
                          indices.end());

            halo_receive_counts_[rank] =
                checked_int(indices.size(), "Halo receive count");
            halo_receive_displacements_[rank] =
                checked_int(total_receive_count, "Halo receive displacement");
            total_receive_count += indices.size();
        }

        checked_int(total_receive_count, "Total halo receive count");

        std::vector<Index> receive_global_indices;
        receive_global_indices.reserve(total_receive_count);

        for (const auto& indices : requested_indices)
        {
            receive_global_indices.insert(receive_global_indices.end(),
                                          indices.begin(),
                                          indices.end());
        }

        std::vector<WireIndex> outgoing_requests(total_receive_count, 0);

        for (Index i = 0; i < total_receive_count; ++i)
        {
            outgoing_requests[i] = to_wire_index(receive_global_indices[i]);
        }

        halo_send_counts_.assign(comm_size, 0);
        MPI_Alltoall(halo_receive_counts_.data(),
                     1,
                     MPI_INT,
                     halo_send_counts_.data(),
                     1,
                     MPI_INT,
                     comm_);

        halo_send_displacements_.assign(comm_size, 0);
        Index total_send_count = 0;

        for (int rank = 0; rank < comm_size; ++rank)
        {
            halo_send_displacements_[rank] =
                checked_int(total_send_count, "Halo send displacement");
            total_send_count += static_cast<Index>(halo_send_counts_[rank]);
        }

        checked_int(total_send_count, "Total halo send count");

        std::vector<WireIndex> incoming_requests(total_send_count, 0);

        MPI_Alltoallv(outgoing_requests.data(),
                      halo_receive_counts_.data(),
                      halo_receive_displacements_.data(),
                      MPI_UNSIGNED_LONG_LONG,
                      incoming_requests.data(),
                      halo_send_counts_.data(),
                      halo_send_displacements_.data(),
                      MPI_UNSIGNED_LONG_LONG,
                      comm_);

        halo_send_local_indices_.resize(total_send_count);

        for (Index i = 0; i < total_send_count; ++i)
        {
            const Index global_index =
                static_cast<Index>(incoming_requests[i]);

            if (global_index < x.local_start() ||
                global_index >= x.local_end())
            {
                throw std::runtime_error(
                    "Received a halo request for a non-local vector entry.");
            }

            halo_send_local_indices_[i] = global_index - x.local_start();
        }

        halo_receive_values_.assign(total_receive_count, 0.0);
        halo_send_values_.assign(total_send_count, 0.0);
        halo_requests_.reserve(static_cast<Index>(comm_size) * 2);

        std::unordered_map<Index, Index> halo_positions;
        halo_positions.reserve(total_receive_count);

        for (Index i = 0; i < total_receive_count; ++i)
        {
            halo_positions.emplace(receive_global_indices[i], i);
        }

        column_value_indices_.resize(col_indices_.size());
        column_uses_local_value_.resize(col_indices_.size());

        for (Index i = 0; i < col_indices_.size(); ++i)
        {
            const Index column = col_indices_[i];

            if (column >= x.local_start() && column < x.local_end())
            {
                column_uses_local_value_[i] = 1;
                column_value_indices_[i] = column - x.local_start();
            }
            else
            {
                const auto position = halo_positions.find(column);

                if (position == halo_positions.end())
                {
                    throw std::runtime_error(
                        "Halo plan is missing a required vector entry.");
                }

                column_uses_local_value_[i] = 0;
                column_value_indices_[i] = position->second;
            }
        }

        halo_local_start_ = x.local_start();
        halo_local_size_ = x.local_size();
        halo_plan_initialized_ = true;
    }

    DistributedVector DistributedSparseMatrixCSR::multiply(const DistributedVector& x) const
    {
        if (x.global_size() != global_cols_)
        {
            throw std::invalid_argument("Distributed vector size must match matrix column count.");
        }

        if (x.communicator() != comm_)
        {
            throw std::invalid_argument("Matrix and vector must use the same MPI communicator.");
        }

        if (!halo_plan_initialized_)
        {
            initialize_halo_plan(x);
        }
        else if (x.local_start() != halo_local_start_ ||
                 x.local_size() != halo_local_size_)
        {
            throw std::invalid_argument(
                "Distributed vector layout differs from the cached halo plan.");
        }

        int comm_size = 0;
        MPI_Comm_size(comm_, &comm_size);

        for (Index i = 0; i < halo_send_values_.size(); ++i)
        {
            halo_send_values_[i] =
                x.local_values()[halo_send_local_indices_[i]];
        }

        halo_requests_.clear();

        for (int rank = 0; rank < comm_size; ++rank)
        {
            if (halo_receive_counts_[rank] == 0)
            {
                continue;
            }

            MPI_Request request = MPI_REQUEST_NULL;
            MPI_Irecv(halo_receive_values_.data() +
                          halo_receive_displacements_[rank],
                      halo_receive_counts_[rank],
                      MPI_DOUBLE,
                      rank,
                      halo_exchange_tag,
                      comm_,
                      &request);
            halo_requests_.push_back(request);
        }

        for (int rank = 0; rank < comm_size; ++rank)
        {
            if (halo_send_counts_[rank] == 0)
            {
                continue;
            }

            MPI_Request request = MPI_REQUEST_NULL;
            MPI_Isend(halo_send_values_.data() +
                          halo_send_displacements_[rank],
                      halo_send_counts_[rank],
                      MPI_DOUBLE,
                      rank,
                      halo_exchange_tag,
                      comm_,
                      &request);
            halo_requests_.push_back(request);
        }

        Vector local_y(local_rows_, 0.0);

        for (Index local_row = 0; local_row < local_rows_; ++local_row)
        {
            for (Index k = row_ptr_[local_row]; k < row_ptr_[local_row + 1]; ++k)
            {
                if (column_uses_local_value_[k])
                {
                    local_y[local_row] +=
                        values_[k] * x.local_values()[column_value_indices_[k]];
                }
            }
        }

        if (!halo_requests_.empty())
        {
            MPI_Waitall(static_cast<int>(halo_requests_.size()),
                        halo_requests_.data(),
                        MPI_STATUSES_IGNORE);
        }

        for (Index local_row = 0; local_row < local_rows_; ++local_row)
        {
            for (Index k = row_ptr_[local_row]; k < row_ptr_[local_row + 1]; ++k)
            {
                if (!column_uses_local_value_[k])
                {
                    local_y[local_row] +=
                        values_[k] *
                        halo_receive_values_[column_value_indices_[k]];
                }
            }
        }

        return DistributedVector(global_rows_,
                                 local_row_start_,
                                 local_y,
                                 comm_);
    }
}
