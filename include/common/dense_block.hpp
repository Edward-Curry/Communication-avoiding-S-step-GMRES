#ifndef COMMON_DENSE_BLOCK_HPP
#define COMMON_DENSE_BLOCK_HPP

#include "common/types.hpp"

namespace gmres {

class DenseBlock {
public:
    DenseBlock() = default;
    DenseBlock(Index rows, Index cols, Scalar value = 0.0);

    Index rows() const;
    Index cols() const;
    Index size() const;

    Scalar* data();
    const Scalar* data() const;

    Scalar& operator()(Index row, Index col);
    const Scalar& operator()(Index row, Index col) const;

    Scalar* column(Index col);
    const Scalar* column(Index col) const;

    void set_column(Index col, const Vector& values);
    Vector get_column(Index col) const;

    DenseBlock leading_columns(Index count) const;
    DenseBlock leading_principal_block(Index count) const;

private:
    Index rows_ = 0;
    Index cols_ = 0;
    Vector values_;
};

DenseBlock transpose_multiply(const DenseBlock& left,
                              const DenseBlock& right);

// Uses only the leading left_cols columns of left, without copying them.
DenseBlock transpose_multiply(const DenseBlock& left,
                              Index left_cols,
                              const DenseBlock& right);

DenseBlock multiply(const DenseBlock& left,
                    const DenseBlock& right);

DenseBlock gram_matrix(const DenseBlock& block);

void subtract_product(DenseBlock& target,
                      const DenseBlock& left,
                      const DenseBlock& right);

// target -= left(:, 0:left_cols) * right, without copying the leading columns.
void subtract_product(DenseBlock& target,
                      const DenseBlock& left,
                      Index left_cols,
                      const DenseBlock& right);

// target += block(:, 0:cols) * coefficients.
void multiply_add_columns(const DenseBlock& block,
                          Index cols,
                          const Vector& coefficients,
                          Vector& target);

void add_in_place(DenseBlock& target,
                  const DenseBlock& source);

void right_solve_upper(DenseBlock& block,
                       const DenseBlock& upper_triangular);

DenseBlock pack_columns(const VectorList& columns);
VectorList unpack_columns(const DenseBlock& block);
DenseMatrix to_dense_matrix(const DenseBlock& block);

}

#endif
