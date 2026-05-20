#ifndef COMMON_SPARSE_MATRIX_HPP
#define COMMON_SPARSE_MATRIX_HPP

#include "common/types.hpp"

#include <vector>

namespace gmres
{
    class SparseMatrixCSR
    {
    public:
        SparseMatrixCSR();

        SparseMatrixCSR(Index rows,
                        Index cols,
                        const Vector& values,
                        const std::vector<Index>& col_indices,
                        const std::vector<Index>& row_ptr);

        Index rows() const;
        Index cols() const;
        Index nonzeros() const;

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