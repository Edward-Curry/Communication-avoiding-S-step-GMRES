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
        void initialize_halo_plan(const DistributedVector& x) const;

        Index global_rows_ = 0;
        Index global_cols_ = 0;

        Index local_row_start_ = 0;
        Index local_rows_ = 0;

        Vector values_;
        std::vector<Index> col_indices_;
        std::vector<Index> row_ptr_;

        MPI_Comm comm_ = MPI_COMM_WORLD;

        mutable bool halo_plan_initialized_ = false;
        mutable Index halo_local_start_ = 0;
        mutable Index halo_local_size_ = 0;

        mutable std::vector<int> halo_receive_counts_;
        mutable std::vector<int> halo_receive_displacements_;
        mutable std::vector<int> halo_send_counts_;
        mutable std::vector<int> halo_send_displacements_;
        mutable std::vector<Index> halo_send_local_indices_;
        mutable Vector halo_receive_values_;
        mutable Vector halo_send_values_;
        mutable std::vector<MPI_Request> halo_requests_;

        mutable std::vector<Index> column_value_indices_;
        mutable std::vector<unsigned char> column_uses_local_value_;
    };
}

#endif
