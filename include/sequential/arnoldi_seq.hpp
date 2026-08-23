/**
 * @file include/sequential/arnoldi_seq.hpp
 * @brief Declares sequential Arnoldi iteration.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef SEQUENTIAL_ARNOLDI_SEQ_HPP
#define SEQUENTIAL_ARNOLDI_SEQ_HPP

#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

namespace gmres
{
    /**
     * @brief Stores an Arnoldi basis and Hessenberg relation.
     */
    struct ArnoldiResult
    {
        VectorList V;
        DenseMatrix H;
        Scalar beta = 0.0;
    };

    /**
     * @brief Generates an m-step Arnoldi factorisation.
     * @param A System matrix.
     * @param r0 Initial residual.
     * @param m Number of Arnoldi steps.
     * @return Basis vectors, Hessenberg matrix, and initial residual norm.
     */
    ArnoldiResult arnoldi_seq(const SparseMatrixCSR& A,
                              const Vector& r0,
                              Index m);
}

#endif
