#ifndef PARALLEL_DISTRIBUTED_VECTOR_OPS_HPP
#define PARALLEL_DISTRIBUTED_VECTOR_OPS_HPP

#include "common/types.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    void check_compatible(const DistributedVector& x,
                          const DistributedVector& y);

    Scalar dot_mpi(const DistributedVector& x,
                   const DistributedVector& y);

    Scalar norm2_mpi(const DistributedVector& x);

    void scal_local(Scalar alpha,
                    DistributedVector& x);

    void axpy_local(Scalar alpha,
                    const DistributedVector& x,
                    DistributedVector& y);
}

#endif

