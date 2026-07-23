#ifndef COMMUNICATION_AVOIDING_HESSENBERG_ASSEMBLY_HPP
#define COMMUNICATION_AVOIDING_HESSENBERG_ASSEMBLY_HPP

#include "common/types.hpp"

namespace gmres {

void append_monomial_hessenberg_block(DenseMatrix& hessenberg,
                                       const DenseMatrix& r_old,
                                       const DenseMatrix& r_block);

// Generalizes append_monomial_hessenberg_block to the shifted recurrence
// A*w_{col-1} = scale[col]*w_col + shifts[col]*w_{col-1} (w_{-1} := q_prev,
// the seed vector). shifts.size() must equal r_block's block size. scales
// may be empty (treated as all-ones, i.e. plain Newton); when non-empty it
// must also match the block size (ScaledNewton). Passing shifts of all zero
// and empty scales reproduces append_monomial_hessenberg_block exactly - the
// monomial recurrence A*w_{col-1} = w_col is the shifts=0/scales=1 case of
// this same relation. The only change from the monomial derivation is how
// the initial right-hand side is built; the "remove already-known old-basis
// contributions" step, the change-of-coordinates matrix, and the forward
// substitution do not depend on the recurrence and are unchanged.
void append_shifted_hessenberg_block(DenseMatrix& hessenberg,
                                     const DenseMatrix& r_old,
                                     const DenseMatrix& r_block,
                                     const Vector& shifts,
                                     const Vector& scales = Vector());

}

#endif
