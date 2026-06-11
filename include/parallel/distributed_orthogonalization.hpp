#ifndef PARALLEL_DISTRIBUTED_ORTHOGONALIZATION_HPP
#define PARALLEL_DISTRIBUTED_ORTHOGONALIZATION_HPP

#include "common/types.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    void modified_gram_schmidt_mpi(const DistributedVectorList& basis,
                                   DistributedVector& w,
                                   Vector& h);
}

#endif