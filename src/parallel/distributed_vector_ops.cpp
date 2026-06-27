#include "parallel/distributed_vector_ops.hpp"

#include <cblas.h>

#include <cmath>
#include <limits>
#include <mpi.h>
#include <stdexcept>

namespace gmres
{
    namespace
    {
        int blas_size(Index size)
        {
            if (size > static_cast<Index>(std::numeric_limits<int>::max()))
            {
                throw std::length_error("Local vector is too large for the configured BLAS integer type.");
            }

            return static_cast<int>(size);
        }
    }

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

        const Scalar local_dot = x_local.empty()
            ? 0.0
            : cblas_ddot(blas_size(x_local.size()),
                          x_local.data(),
                          1,
                          y_local.data(),
                          1);

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

        if (x_local.empty())
        {
            return;
        }

        cblas_dscal(blas_size(x_local.size()),
                    alpha,
                    x_local.data(),
                    1);
    }

    void axpy_local(Scalar alpha,
                    const DistributedVector& x,
                    DistributedVector& y)
    {
        check_compatible(x, y);

        const Vector& x_local = x.local_values();
        Vector& y_local = y.local_values();

        if (x_local.empty())
        {
            return;
        }

        cblas_daxpy(blas_size(x_local.size()),
                    alpha,
                    x_local.data(),
                    1,
                    y_local.data(),
                    1);
    }
}
