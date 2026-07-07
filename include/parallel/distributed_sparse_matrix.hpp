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
        struct HaloExchangePlan
        {
            bool initialized = false;
            Index vector_local_start = 0;
            Index vector_local_size = 0;

            std::vector<int> recv_ranks;
            std::vector<int> recv_counts;
            std::vector<int> recv_displacements;
            std::vector<int> send_ranks;
            std::vector<int> send_counts;
            std::vector<int> send_displacements;

            std::vector<Index> send_local_indices;
            std::vector<Index> nonzero_x_indices;
            Index remote_value_count = 0;

            Vector send_values;
            Vector remote_values;
            std::vector<MPI_Request> requests;
        };

        void initialize_halo_plan(const DistributedVector& x) const;

        Index global_rows_ = 0;
        Index global_cols_ = 0;

        Index local_row_start_ = 0;
        Index local_rows_ = 0;

        Vector values_;
        std::vector<Index> col_indices_;
        std::vector<Index> row_ptr_;

        MPI_Comm comm_ = MPI_COMM_WORLD;
        mutable HaloExchangePlan halo_plan_;
    };
}

#endif
