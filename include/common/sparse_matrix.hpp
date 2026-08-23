/**
 * @file include/common/sparse_matrix.hpp
 * @brief Declares a compressed sparse row matrix.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMON_SPARSE_MATRIX_HPP
#define COMMON_SPARSE_MATRIX_HPP

#include "common/types.hpp"

#include <vector>

namespace gmres
{
    /**
     * @brief Stores a sparse matrix in compressed sparse row format.
     */
    class SparseMatrixCSR
    {
    public:
        /** @brief Constructs an empty sparse matrix. */
        SparseMatrixCSR();

        /**
         * @brief Constructs a sparse matrix from CSR arrays.
         * @param rows Matrix row count.
         * @param cols Matrix column count.
         * @param values Nonzero values.
         * @param col_indices Column index for each nonzero.
         * @param row_ptr Row offsets with rows plus one entries.
         */
        SparseMatrixCSR(Index rows,
                        Index cols,
                        const Vector& values,
                        const std::vector<Index>& col_indices,
                        const std::vector<Index>& row_ptr);

        /** @return Matrix row count. */
        Index rows() const;
        /** @return Matrix column count. */
        Index cols() const;
        /** @return Number of stored nonzeros. */
        Index nonzeros() const;

        /**
         * @brief Computes a sparse matrix-vector product.
         * @param x Vector with cols() entries.
         * @return Product of this matrix and x.
         */
        Vector multiply(const Vector& x) const;

    private:
        Index rows_;
        Index cols_;

        Vector values_;
        std::vector<Index> col_indices_;
        std::vector<Index> row_ptr_;
    };
}

#endif
