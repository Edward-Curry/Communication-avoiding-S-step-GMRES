/**
 * @file src/common/vector_ops.cpp
 * @brief Implements dense vector operations.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "common/vector_ops.hpp"

#include <cblas.h>

#include <limits>
#include <stdexcept>

namespace gmres
{
    namespace
    {
        /**
         * @brief Converts a project dimension to the BLAS integer type.
         * @param size Dimension to convert.
         * @return BLAS-compatible dimension.
         */
        int blas_size(Index size)
        {
            if (size > static_cast<Index>(std::numeric_limits<int>::max()))
            {
                throw std::length_error("Vector is too large for the configured BLAS integer type.");
            }

            return static_cast<int>(size);
        }
    }

    void check_same_size(const Vector& x, const Vector& y)
    {
        if (x.size() != y.size())
        {
            throw std::invalid_argument("Vector sizes do not match.");
        }
    }

    Scalar dot(const Vector& x, const Vector& y)
    {
        check_same_size(x, y);

        if (x.empty())
        {
            return 0.0;
        }

        return cblas_ddot(blas_size(x.size()),
                          x.data(),
                          1,
                          y.data(),
                          1);
    }

    Scalar norm2(const Vector& x)
    {
        if (x.empty())
        {
            return 0.0;
        }

        return cblas_dnrm2(blas_size(x.size()), x.data(), 1);
    }

    void scal(Scalar alpha, Vector& x)
    {
        if (x.empty())
        {
            return;
        }

        cblas_dscal(blas_size(x.size()), alpha, x.data(), 1);
    }

    void axpy(Scalar alpha, const Vector& x, Vector& y)
    {
        check_same_size(x, y);

        if (x.empty())
        {
            return;
        }

        cblas_daxpy(blas_size(x.size()),
                    alpha,
                    x.data(),
                    1,
                    y.data(),
                    1);
    }
}
