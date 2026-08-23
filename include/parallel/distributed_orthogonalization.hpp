/**
 * @file include/parallel/distributed_orthogonalization.hpp
 * @brief Declares MPI modified Gram-Schmidt.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef PARALLEL_DISTRIBUTED_ORTHOGONALIZATION_HPP
#define PARALLEL_DISTRIBUTED_ORTHOGONALIZATION_HPP

#include "common/types.hpp"
#include "parallel/distributed_vector.hpp"

namespace gmres
{
    /**
     * @brief Orthogonalises and normalises a distributed vector.
     * @param basis Existing distributed orthonormal basis.
     * @param w Vector replaced by its normalized orthogonal component.
     * @param h Replicated projection coefficients and final norm.
     */
    void modified_gram_schmidt_mpi(const DistributedVectorList& basis,
                                   DistributedVector& w,
                                   Vector& h);
}

#endif
