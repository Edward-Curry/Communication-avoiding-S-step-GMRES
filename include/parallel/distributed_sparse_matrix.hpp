#ifndef PARALLEL_DISTRIBUTED_SPARSE_MATRIX_HPP
#define PARALLEL_DISTRIBUTED_SPARSE_MATRIX_HPP

#include "common/types.hpp"
#include "parallel/distributed_vector.hpp"

#include <mpi.h>
#include <vector>

namespace gmres
{
    class DistributedSparseMatrixCSR
    {
    public:
        DistributedSparseMatrixCSR();

        DistributedSparseMatrixCSR(Index global_rows,
                                   Index global_cols,
                                   Index local_row_start,
                                   Index local_rows,
                                   const Vector& values,
                                   const std::vector<Index>& col_indices,
                                   const std::vector<Index>& row_ptr,
                                   MPI_Comm comm);

        Index global_rows() const;
        Index global_cols() const;

        Index local_row_start() const;
        Index local_rows() const;
        Index local_row_end() const;

        Index nonzeros() const;

        MPI_Comm communicator() const;

        DistributedVector multiply(const DistributedVector& x) const;

    private:
        Index global_rows_ = 0;
        Index global_cols_ = 0;

        Index local_row_start_ = 0;
        Index local_rows_ = 0;

        Vector values_;
        std::vector<Index> col_indices_;
        std::vector<Index> row_ptr_;

        MPI_Comm comm_ = MPI_COMM_WORLD;
    };
}

#endif