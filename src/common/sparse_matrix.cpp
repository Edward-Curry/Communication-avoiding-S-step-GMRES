#include "common/sparse_matrix.hpp"

#include <stdexcept>

namespace gmres
{
    SparseMatrixCSR::SparseMatrixCSR()
        : rows_(0),
          cols_(0)
    {
    }

    SparseMatrixCSR::SparseMatrixCSR(Index rows,
                                     Index cols,
                                     const Vector& values,
                                     const std::vector<Index>& col_indices,
                                     const std::vector<Index>& row_ptr)
        : rows_(rows),
          cols_(cols),
          values_(values),
          col_indices_(col_indices),
          row_ptr_(row_ptr)
    {
        if (row_ptr_.size() != rows_ + 1)
        {
            throw std::invalid_argument("row_ptr size must be rows + 1.");
        }

        if (values_.size() != col_indices_.size())
        {
            throw std::invalid_argument("values and col_indices must have the same size.");
        }

        if (!row_ptr_.empty() && row_ptr_.back() != values_.size())
        {
            throw std::invalid_argument("Last row_ptr entry must equal number of nonzeros.");
        }
    }

    Index SparseMatrixCSR::rows() const
    {
        return rows_;
    }

    Index SparseMatrixCSR::cols() const
    {
        return cols_;
    }

    Index SparseMatrixCSR::nonzeros() const
    {
        return values_.size();
    }

    Vector SparseMatrixCSR::multiply(const Vector& x) const
    {
        if (x.size() != cols_)
        {
            throw std::invalid_argument("Input vector size does not match matrix columns.");
        }

        Vector y(rows_, 0.0);

        for (Index i = 0; i < rows_; ++i)
        {
            for (Index k = row_ptr_[i]; k < row_ptr_[i + 1]; ++k)
            {
                y[i] += values_[k] * x[col_indices_[k]];
            }
        }

        return y;
    }
}