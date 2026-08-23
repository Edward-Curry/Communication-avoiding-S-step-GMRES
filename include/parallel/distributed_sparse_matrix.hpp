/**
 * @file include/parallel/distributed_sparse_matrix.hpp
 * @brief Declares a row-distributed CSR matrix with halo exchange.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_DISTRIBUTED_SPARSE_MATRIX_HPP
#define PARALLEL_DISTRIBUTED_SPARSE_MATRIX_HPP

#include "common/types.hpp"
#include "parallel/distributed_vector.hpp"

#include <mpi.h>
#include <vector>

namespace gmres
{
    /**
     * @brief Reports cached point-to-point halo exchange sizes.
     */
    struct HaloExchangeStatistics
    {
        Index receive_peer_count = 0;
        Index send_peer_count = 0;
        Index receive_value_count = 0;
        Index send_value_count = 0;
    };

    /**
     * @brief Stores a CSR matrix partitioned by contiguous rows.
     */
    class DistributedSparseMatrixCSR
    {
    public:
        /** @brief Constructs an empty distributed sparse matrix. */
        DistributedSparseMatrixCSR();

        /**
         * @brief Constructs a distributed matrix from local CSR arrays.
         * @param global_rows Global matrix row count.
         * @param global_cols Global matrix column count.
         * @param local_row_start Global index of the first local row.
         * @param local_rows Number of locally owned rows.
         * @param values Local nonzero values.
         * @param col_indices Global column index for each local nonzero.
         * @param row_ptr Local CSR row offsets.
         * @param comm Communicator defining the partition.
         */
        DistributedSparseMatrixCSR(Index global_rows,
                                   Index global_cols,
                                   Index local_row_start,
                                   Index local_rows,
                                   const Vector& values,
                                   const std::vector<Index>& col_indices,
                                   const std::vector<Index>& row_ptr,
                                   MPI_Comm comm);

        /** @return Global row count. */
        Index global_rows() const;
        /** @return Global column count. */
        Index global_cols() const;

        /** @return Global index of the first local row. */
        Index local_row_start() const;
        /** @return Number of locally owned rows. */
        Index local_rows() const;
        /** @return Exclusive global end index of local rows. */
        Index local_row_end() const;

        /** @return Number of locally stored nonzeros. */
        Index nonzeros() const;

        /** @return Communicator defining the matrix partition. */
        MPI_Comm communicator() const;

        /**
         * @brief Returns counts from the cached halo schedule.
         * @return Local receive and send peer and value counts.
         */
        HaloExchangeStatistics halo_exchange_statistics() const;

        /**
         * @brief Computes a distributed sparse matrix-vector product.
         * @param x Distributed input vector with compatible row partition.
         * @return Distributed product vector.
         */
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

        /**
         * @brief Builds the cached halo schedule for a compatible vector layout.
         * @param x Distributed vector defining the requested column values.
         */
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
