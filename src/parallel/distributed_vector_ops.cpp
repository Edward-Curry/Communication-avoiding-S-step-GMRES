#include "parallel/distributed_vector_ops.hpp"

#include <cmath>
#include <mpi.h>
#include <stdexcept>

namespace gmres
{
    void check_compatible(const DistributedVector& x,
                          const DistributedVector& y)
    {
        if (x.global_size() != y.global_size())
        {
            throw std::invalid_argument("Distributed vectors have different global sizes.");
        }

        if (x.local_size() != y.local_size())
        {
            throw std::invalid_argument("Distributed vectors have different local sizes.");
        }

        if (x.local_start() != y.local_start())
        {
            throw std::invalid_argument("Distributed vectors have different local starts.");
        }

        if (x.communicator() != y.communicator())
        {
            throw std::invalid_argument("Distributed vectors use different MPI communicators.");
        }
    }

    Scalar dot_mpi(const DistributedVector& x,
                   const DistributedVector& y)
    {
        check_compatible(x, y);

        const Vector& x_local = x.local_values();
        const Vector& y_local = y.local_values();

        Scalar local_dot = 0.0;

        for (Index i = 0; i < x_local.size(); ++i)
        {
            local_dot += x_local[i] * y_local[i];
        }

        Scalar global_dot = 0.0;

        MPI_Allreduce(&local_dot,
                      &global_dot,
                      1,
                      MPI_DOUBLE,
                      MPI_SUM,
                      x.communicator());

        return global_dot;
    }

    Scalar norm2_mpi(const DistributedVector& x)
    {
        return std::sqrt(dot_mpi(x, x));
    }

    void scal_local(Scalar alpha,
                    DistributedVector& x)
    {
        Vector& x_local = x.local_values();

        for (Index i = 0; i < x_local.size(); ++i)
        {
            x_local[i] *= alpha;
        }
    }

    void axpy_local(Scalar alpha,
                    const DistributedVector& x,
                    DistributedVector& y)
    {
        check_compatible(x, y);

        const Vector& x_local = x.local_values();
        Vector& y_local = y.local_values();

        for (Index i = 0; i < x_local.size(); ++i)
        {
            y_local[i] += alpha * x_local[i];
        }
    }
}
