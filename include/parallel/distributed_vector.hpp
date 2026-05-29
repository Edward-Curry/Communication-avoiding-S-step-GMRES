#ifndef PARALLEL_DISTRIBUTED_VECTOR_HPP
#define PARALLEL_DISTRIBUTED_VECTOR_HPP

#include "common/types.hpp"

#include <mpi.h>

namespace gmres
{
    class DistributedVector
    {
    public:
        DistributedVector();

        DistributedVector(Index global_size,
                          Index local_start,
                          Vector local_values,
                          MPI_Comm comm);

        Index global_size() const;
        Index local_size() const;
        Index local_start() const;
        Index local_end() const;

        MPI_Comm communicator() const;

        const Vector& local_values() const;
        Vector& local_values();

    private:
        Index global_size_ = 0;
        Index local_start_ = 0;
        Vector local_values_;
        MPI_Comm comm_ = MPI_COMM_WORLD;
    };
}

#endif