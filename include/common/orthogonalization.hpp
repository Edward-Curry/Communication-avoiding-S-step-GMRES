/**
 * @file include/common/orthogonalization.hpp
 * @brief Declares sequential orthogonalisation routines.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMON_ORTHOGONALIZATION_HPP
#define COMMON_ORTHOGONALIZATION_HPP

#include "common/types.hpp"

#include <vector>

namespace gmres
{
    /**
     * @brief Orthogonalises and normalises a vector against a basis.
     * @param basis Existing orthonormal basis vectors.
     * @param w Vector replaced by its normalized orthogonal component.
     * @param h Output projection coefficients and final norm.
     */
    void modified_gram_schmidt(const std::vector<Vector>& basis,
                               Vector& w,
                               Vector& h);
}

#endif
