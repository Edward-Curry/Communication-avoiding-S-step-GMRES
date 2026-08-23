/**
 * @file src/parallel/distributed_orthogonalization.cpp
 * @brief Implements distributed modified Gram-Schmidt orthogonalization.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "parallel/distributed_orthogonalization.hpp"

#include "parallel/distributed_vector_ops.hpp"

#include <stdexcept>

namespace gmres
{
    void modified_gram_schmidt_mpi(const DistributedVectorList& basis,
                                   DistributedVector& w,
                                   Vector& h)
    {
        h.clear();
        h.reserve(basis.size() + 1);

        for (const DistributedVector& v : basis)
        {
            Scalar coefficient = dot_mpi(v, w);
            h.push_back(coefficient);

            axpy_local(-coefficient, v, w);
        }

        Scalar remaining_norm = norm2_mpi(w);
        h.push_back(remaining_norm);

        if (remaining_norm == 0.0)
        {
            throw std::runtime_error("Breakdown in MPI modified Gram-Schmidt: zero remaining vector.");
        }

        scal_local(1.0 / remaining_norm, w);
    }
}
