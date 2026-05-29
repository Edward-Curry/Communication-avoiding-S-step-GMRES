#include "parallel/distributed_vector.hpp"

#include <stdexcept>
#include <utility>

namespace gmres
{
    DistributedVector::DistributedVector()
        : global_size_(0),
          local_start_(0),
          local_values_(),
          comm_(MPI_COMM_WORLD)
    {
    }

    DistributedVector::DistributedVector(Index global_size,
                                         Index local_start,
                                         Vector local_values,
                                         MPI_Comm comm)
        : global_size_(global_size),
          local_start_(local_start),
          local_values_(std::move(local_values)),
          comm_(comm)
    {
        if (local_start_ > global_size_)
        {
            throw std::invalid_argument("DistributedVector local_start exceeds global_size.");
        }

        if (local_start_ + local_values_.size() > global_size_)
        {
            throw std::invalid_argument("DistributedVector local range exceeds global_size.");
        }
    }

    Index DistributedVector::global_size() const
    {
        return global_size_;
    }

    Index DistributedVector::local_size() const
    {
        return local_values_.size();
    }

    Index DistributedVector::local_start() const
    {
        return local_start_;
    }

    Index DistributedVector::local_end() const
    {
        return local_start_ + local_values_.size();
    }

    MPI_Comm DistributedVector::communicator() const
    {
        return comm_;
    }

    const Vector& DistributedVector::local_values() const
    {
        return local_values_;
    }

    Vector& DistributedVector::local_values()
    {
        return local_values_;
    }
}