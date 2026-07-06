#ifndef COMMUNICATION_AVOIDING_HESSENBERG_ASSEMBLY_HPP
#define COMMUNICATION_AVOIDING_HESSENBERG_ASSEMBLY_HPP

#include "common/types.hpp"

namespace gmres {

void append_monomial_hessenberg_block(DenseMatrix& hessenberg,
                                       const DenseMatrix& r_old,
                                       const DenseMatrix& r_block);

}

#endif
