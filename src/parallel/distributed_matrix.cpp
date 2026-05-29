#include "parallel/distributed_sparse_matrix.hpp"

#include <stdexcept>
#include <vector>

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

        int comm_size = 0;
        int comm_rank = 0;

        MPI_Comm_size(comm_, &comm_size);
        MPI_Comm_rank(comm_, &comm_rank);

        int local_x_size = static_cast<int>(x.local_size());

        std::vector<int> recv_counts(comm_size, 0);
        MPI_Allgather(&local_x_size,
                      1,
                      MPI_INT,
                      recv_counts.data(),
                      1,
                      MPI_INT,
                      comm_);

        std::vector<int> displacements(comm_size, 0);

        for (int rank = 1; rank < comm_size; ++rank)
        {
            displacements[rank] = displacements[rank - 1] + recv_counts[rank - 1];
        }

        int total_size = displacements[comm_size - 1] + recv_counts[comm_size - 1];

        if (static_cast<Index>(total_size) != global_cols_)
        {
            throw std::runtime_error("Distributed vector pieces do not add up to global column count.");
        }

        Vector full_x(global_cols_, 0.0);

        MPI_Allgatherv(x.local_values().data(),
                       local_x_size,
                       MPI_DOUBLE,
                       full_x.data(),
                       recv_counts.data(),
                       displacements.data(),
                       MPI_DOUBLE,
                       comm_);

        Vector local_y(local_rows_, 0.0);

        for (Index local_row = 0; local_row < local_rows_; ++local_row)
        {
            for (Index k = row_ptr_[local_row]; k < row_ptr_[local_row + 1]; ++k)
            {
                local_y[local_row] += values_[k] * full_x[col_indices_[k]];
            }
        }

        return DistributedVector(global_rows_,
                                 local_row_start_,
                                 local_y,
                                 comm_);
    }
}