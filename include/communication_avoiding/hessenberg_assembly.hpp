/**
 * @file include/communication_avoiding/hessenberg_assembly.hpp
 * @brief Declares CA-GMRES Hessenberg assembly.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_HESSENBERG_ASSEMBLY_HPP
#define COMMUNICATION_AVOIDING_HESSENBERG_ASSEMBLY_HPP

#include "common/types.hpp"

namespace gmres {

/**
 * @brief Appends one monomial s-step recurrence block.
 * @param hessenberg Hessenberg matrix updated in place.
 * @param r_old Coefficients against the previous basis.
 * @param r_block Intra-block triangular coefficients.
 */
void append_monomial_hessenberg_block(DenseMatrix& hessenberg,
                                       const DenseMatrix& r_old,
                                       const DenseMatrix& r_block);

/**
 * @brief Appends one shifted Newton recurrence block.
 * @param hessenberg Hessenberg matrix updated in place.
 * @param r_old Coefficients against the previous basis.
 * @param r_block Intra-block triangular coefficients.
 * @param shifts Newton shifts for accepted columns.
 * @param scales Optional Scaled-Newton factors.
 */
void append_shifted_hessenberg_block(DenseMatrix& hessenberg,
                                     const DenseMatrix& r_old,
                                     const DenseMatrix& r_block,
                                     const Vector& shifts,
                                     const Vector& scales = Vector());

}

#endif
