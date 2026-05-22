#ifndef SEQUENTIAL_ARNOLDI_SEQ_HPP
#define SEQUENTIAL_ARNOLDI_SEQ_HPP

#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

namespace gmres
{
    struct ArnoldiResult
    {
        VectorList V;
        DenseMatrix H;
        Scalar beta = 0.0;
    };

    ArnoldiResult arnoldi_seq(const SparseMatrixCSR& A,
                              const Vector& r0,
                              Index m);
}

#endif