/**
 * @file include/parallel/distributed_io.hpp
 * @brief Declares distributed Matrix Market I/O.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_DISTRIBUTED_IO_HPP
#define PARALLEL_DISTRIBUTED_IO_HPP

#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

#include <mpi.h>
#include <string>

namespace gmres
{
    /**
     * @brief Reads and row-distributes a Matrix Market sparse matrix.
     * @param filename Matrix Market file path.
     * @param comm Communicator used for the distribution.
     * @return Distributed CSR matrix.
     */
    DistributedSparseMatrixCSR read_matrix_market_distributed(const std::string& filename,
                                                              MPI_Comm comm);

    /**
     * @brief Reads and distributes a Matrix Market vector.
     * @param filename Matrix Market vector file path.
     * @param comm Communicator used for the distribution.
     * @return Distributed vector.
     */
    DistributedVector read_matrix_market_vector_distributed(const std::string& filename,
                                                            MPI_Comm comm);

    /**
     * @brief Constructs a distributed vector from global values.
     * @param global_values Full vector available on each caller.
     * @param local_start Global index of the first local value.
     * @param local_rows Number of local values.
     * @param comm Communicator defining the distribution.
     * @return Distributed vector containing the selected local range.
     */
    DistributedVector make_distributed_vector(const Vector& global_values,
                                              Index local_start,
                                              Index local_rows,
                                              MPI_Comm comm);

    /**
     * @brief Gathers a distributed vector to one rank.
     * @param x Distributed input vector.
     * @param comm Communicator defining x.
     * @param root Rank receiving the full vector.
     * @return Full vector on root and an empty vector elsewhere.
     */
    Vector gather_distributed_vector(const DistributedVector& x,
                                     MPI_Comm comm,
                                     int root = 0);
}

#endif
