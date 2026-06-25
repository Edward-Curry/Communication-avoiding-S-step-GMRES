#ifndef PARALLEL_DISTRIBUTED_IO_HPP
#define PARALLEL_DISTRIBUTED_IO_HPP

#include "common/types.hpp"
#include "parallel/distributed_sparse_matrix.hpp"
#include "parallel/distributed_vector.hpp"

#include <mpi.h>
#include <string>

namespace gmres
{
    DistributedSparseMatrixCSR read_matrix_market_distributed(const std::string& filename,
                                                              MPI_Comm comm);

    DistributedVector read_matrix_market_vector_distributed(const std::string& filename,
                                                            MPI_Comm comm);

    DistributedVector make_distributed_vector(const Vector& global_values,
                                              Index local_start,
                                              Index local_rows,
                                              MPI_Comm comm);

    Vector gather_distributed_vector(const DistributedVector& x,
                                     MPI_Comm comm,
                                     int root = 0);
}

#endif
