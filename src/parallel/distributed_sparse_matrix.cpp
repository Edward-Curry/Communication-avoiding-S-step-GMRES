/**
 * @file src/parallel/distributed_sparse_matrix.cpp
 * @brief Implements distributed CSR sparse matrix operations and halo exchange.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "parallel/distributed_sparse_matrix.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
    using WireIndex = std::uint64_t;

    /**
     * @brief Converts an MPI count after checking the MPI integer range.
     * @param value Count to convert.
     * @param description Name used in a range-error message.
     * @return MPI-compatible integer count.
     */
    int checked_int(gmres::Index value, const char* description)
    {
        if (value > static_cast<gmres::Index>(std::numeric_limits<int>::max()))
        {
            throw std::overflow_error(std::string(description) + " exceeds the MPI int count limit.");
        }

        return static_cast<int>(value);
    }
}

namespace gmres
{
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

    HaloExchangeStatistics DistributedSparseMatrixCSR::halo_exchange_statistics() const
    {
        if (!halo_plan_.initialized)
        {
            throw std::logic_error(
                "Halo exchange statistics are unavailable before the first matrix-vector product.");
        }

        return {
            static_cast<Index>(halo_plan_.recv_ranks.size()),
            static_cast<Index>(halo_plan_.send_ranks.size()),
            halo_plan_.remote_value_count,
            static_cast<Index>(halo_plan_.send_local_indices.size())
        };
    }

    void DistributedSparseMatrixCSR::initialize_halo_plan(const DistributedVector& x) const
    {
        int comm_size = 0;
        int comm_rank = 0;
        MPI_Comm_size(comm_, &comm_size);
        MPI_Comm_rank(comm_, &comm_rank);

        const WireIndex local_range[2] = {
            static_cast<WireIndex>(x.local_start()),
            static_cast<WireIndex>(x.local_size())};
        std::vector<WireIndex> all_ranges(static_cast<Index>(2 * comm_size), 0);

        MPI_Allgather(local_range,
                      2,
                      MPI_UNSIGNED_LONG_LONG,
                      all_ranges.data(),
                      2,
                      MPI_UNSIGNED_LONG_LONG,
                      comm_);

        std::vector<Index> starts(comm_size, 0);
        std::vector<Index> ends(comm_size, 0);
        Index expected_start = 0;

        for (int rank = 0; rank < comm_size; ++rank)
        {
            starts[rank] = static_cast<Index>(all_ranges[2 * rank]);
            const Index size = static_cast<Index>(all_ranges[2 * rank + 1]);
            ends[rank] = starts[rank] + size;

            if (starts[rank] != expected_start || ends[rank] > global_cols_)
            {
                throw std::invalid_argument(
                    "Distributed vector ownership must be contiguous, ordered by rank, and cover each global entry once.");
            }

            expected_start = ends[rank];
        }

        if (expected_start != global_cols_)
        {
            throw std::invalid_argument("Distributed vector ownership does not cover the matrix columns.");
        }

        auto owner_of = [&](Index global_index)
        {
            const auto iterator = std::upper_bound(ends.begin(), ends.end(), global_index);
            if (iterator == ends.end())
            {
                throw std::invalid_argument("Matrix column is not owned by any vector rank.");
            }
            return static_cast<int>(iterator - ends.begin());
        };

        std::vector<std::vector<Index>> requests_by_rank(comm_size);
        for (Index column : col_indices_)
        {
            const int owner = owner_of(column);
            if (owner != comm_rank)
            {
                requests_by_rank[owner].push_back(column);
            }
        }

        std::vector<int> request_counts(comm_size, 0);
        std::vector<int> request_displacements(comm_size, 0);
        for (int rank = 0; rank < comm_size; ++rank)
        {
            auto& requests = requests_by_rank[rank];
            std::sort(requests.begin(), requests.end());
            requests.erase(std::unique(requests.begin(), requests.end()), requests.end());
            request_counts[rank] = checked_int(requests.size(), "Halo request count");
            if (rank > 0)
            {
                request_displacements[rank] =
                    request_displacements[rank - 1] + request_counts[rank - 1];
            }
        }

        const int total_requests = std::accumulate(request_counts.begin(), request_counts.end(), 0);
        std::vector<WireIndex> requested_indices(static_cast<Index>(total_requests), 0);
        std::unordered_map<Index, Index> remote_positions;
        remote_positions.reserve(static_cast<Index>(total_requests));

        for (int rank = 0; rank < comm_size; ++rank)
        {
            for (Index i = 0; i < requests_by_rank[rank].size(); ++i)
            {
                const Index position = static_cast<Index>(request_displacements[rank]) + i;
                const Index global_index = requests_by_rank[rank][i];
                requested_indices[position] = static_cast<WireIndex>(global_index);
                remote_positions.emplace(global_index, position);
            }
        }

        std::vector<int> incoming_counts(comm_size, 0);
        MPI_Alltoall(request_counts.data(),
                     1,
                     MPI_INT,
                     incoming_counts.data(),
                     1,
                     MPI_INT,
                     comm_);

        std::vector<int> incoming_displacements(comm_size, 0);
        for (int rank = 1; rank < comm_size; ++rank)
        {
            incoming_displacements[rank] =
                incoming_displacements[rank - 1] + incoming_counts[rank - 1];
        }

        const int total_incoming = std::accumulate(incoming_counts.begin(), incoming_counts.end(), 0);
        std::vector<WireIndex> incoming_indices(static_cast<Index>(total_incoming), 0);

        WireIndex empty_send = 0;
        WireIndex empty_receive = 0;
        MPI_Alltoallv(requested_indices.empty() ? &empty_send : requested_indices.data(),
                      request_counts.data(),
                      request_displacements.data(),
                      MPI_UNSIGNED_LONG_LONG,
                      incoming_indices.empty() ? &empty_receive : incoming_indices.data(),
                      incoming_counts.data(),
                      incoming_displacements.data(),
                      MPI_UNSIGNED_LONG_LONG,
                      comm_);

        HaloExchangePlan plan;
        plan.vector_local_start = x.local_start();
        plan.vector_local_size = x.local_size();
        plan.remote_value_count = static_cast<Index>(total_requests);
        plan.send_local_indices.reserve(static_cast<Index>(total_incoming));

        for (WireIndex wire_index : incoming_indices)
        {
            const Index global_index = static_cast<Index>(wire_index);
            if (global_index < x.local_start() || global_index >= x.local_end())
            {
                throw std::runtime_error("A rank requested a halo entry outside this rank's vector ownership.");
            }
            plan.send_local_indices.push_back(global_index - x.local_start());
        }

        for (int rank = 0; rank < comm_size; ++rank)
        {
            if (request_counts[rank] > 0)
            {
                plan.recv_ranks.push_back(rank);
                plan.recv_counts.push_back(request_counts[rank]);
                plan.recv_displacements.push_back(request_displacements[rank]);
            }
            if (incoming_counts[rank] > 0)
            {
                plan.send_ranks.push_back(rank);
                plan.send_counts.push_back(incoming_counts[rank]);
                plan.send_displacements.push_back(incoming_displacements[rank]);
            }
        }

        plan.nonzero_x_indices.reserve(col_indices_.size());
        for (Index column : col_indices_)
        {
            const int owner = owner_of(column);
            if (owner == comm_rank)
            {
                plan.nonzero_x_indices.push_back(column - x.local_start());
            }
            else
            {
                plan.nonzero_x_indices.push_back(x.local_size() + remote_positions.at(column));
            }
        }

        plan.send_values.resize(plan.send_local_indices.size());
        plan.remote_values.resize(plan.remote_value_count);
        plan.requests.resize(plan.recv_ranks.size() + plan.send_ranks.size(), MPI_REQUEST_NULL);

        plan.initialized = true;
        halo_plan_ = std::move(plan);
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

        if (!halo_plan_.initialized)
        {
            initialize_halo_plan(x);
        }

        if (x.local_start() != halo_plan_.vector_local_start ||
            x.local_size() != halo_plan_.vector_local_size)
        {
            throw std::invalid_argument("Distributed vector partition differs from the matrix's cached halo plan.");
        }

        for (Index i = 0; i < halo_plan_.send_local_indices.size(); ++i)
        {
            halo_plan_.send_values[i] = x.local_values()[halo_plan_.send_local_indices[i]];
        }

        constexpr int halo_tag = 1729;
        Index request_index = 0;
        for (Index i = 0; i < halo_plan_.recv_ranks.size(); ++i)
        {
            MPI_Irecv(halo_plan_.remote_values.data() + halo_plan_.recv_displacements[i],
                      halo_plan_.recv_counts[i],
                      MPI_DOUBLE,
                      halo_plan_.recv_ranks[i],
                      halo_tag,
                      comm_,
                      &halo_plan_.requests[request_index++]);
        }
        for (Index i = 0; i < halo_plan_.send_ranks.size(); ++i)
        {
            MPI_Isend(halo_plan_.send_values.data() + halo_plan_.send_displacements[i],
                      halo_plan_.send_counts[i],
                      MPI_DOUBLE,
                      halo_plan_.send_ranks[i],
                      halo_tag,
                      comm_,
                      &halo_plan_.requests[request_index++]);
        }

        Vector local_y(local_rows_, 0.0);

        // Entries whose vector values are locally owned can be evaluated while
        // remote halo values are in flight.
        for (Index local_row = 0; local_row < local_rows_; ++local_row)
        {
            for (Index k = row_ptr_[local_row]; k < row_ptr_[local_row + 1]; ++k)
            {
                const Index x_index = halo_plan_.nonzero_x_indices[k];
                if (x_index < x.local_size())
                {
                    local_y[local_row] += values_[k] * x.local_values()[x_index];
                }
            }
        }

        if (!halo_plan_.requests.empty())
        {
            MPI_Waitall(static_cast<int>(halo_plan_.requests.size()),
                        halo_plan_.requests.data(),
                        MPI_STATUSES_IGNORE);
        }

        for (Index local_row = 0; local_row < local_rows_; ++local_row)
        {
            for (Index k = row_ptr_[local_row]; k < row_ptr_[local_row + 1]; ++k)
            {
                const Index x_index = halo_plan_.nonzero_x_indices[k];
                if (x_index >= x.local_size())
                {
                    local_y[local_row] +=
                        values_[k] * halo_plan_.remote_values[x_index - x.local_size()];
                }
            }
        }

        return DistributedVector(global_rows_,
                                 local_row_start_,
                                 local_y,
                                 comm_);
    }
}
