/**
 * @file include/common/givens.hpp
 * @brief Declares Givens rotation helpers.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMON_GIVENS_HPP
#define COMMON_GIVENS_HPP

#include "common/types.hpp"

namespace gmres
{
    /**
     * @brief Builds a rotation that eliminates the second input.
     * @param a First input value.
     * @param b Second input value.
     * @param c Output cosine.
     * @param s Output sine.
     */
    void generate_givens(Scalar a,
                         Scalar b,
                         Scalar& c,
                         Scalar& s);

    /**
     * @brief Applies a Givens rotation to two values.
     * @param c Rotation cosine.
     * @param s Rotation sine.
     * @param a First value, replaced by the rotated value.
     * @param b Second value, replaced by the rotated value.
     */
    void apply_givens(Scalar c,
                      Scalar s,
                      Scalar& a,
                      Scalar& b);
}

#endif
