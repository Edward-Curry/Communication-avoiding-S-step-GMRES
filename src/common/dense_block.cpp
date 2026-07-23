#include "common/dense_block.hpp"

#include <cblas.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace gmres {

namespace {

int blas_size(Index size)
{
    if (size > static_cast<Index>(std::numeric_limits<int>::max())) {
        throw std::length_error("Dense block dimension exceeds the configured BLAS integer range.");
    }

    return static_cast<int>(size);
}

void check_same_rows(const DenseBlock& left, const DenseBlock& right)
{
    if (left.rows() != right.rows()) {
        throw std::invalid_argument("Dense blocks have different row counts.");
    }
}

}

DenseBlock::DenseBlock(Index rows, Index cols, Scalar value)
    : rows_(rows), cols_(cols), values_(rows * cols, value)
{
}

Index DenseBlock::rows() const
{
    return rows_;
}

Index DenseBlock::cols() const
{
    return cols_;
}

Index DenseBlock::size() const
{
    return values_.size();
}

Scalar* DenseBlock::data()
{
    return values_.data();
}

const Scalar* DenseBlock::data() const
{
    return values_.data();
}

Scalar& DenseBlock::operator()(Index row, Index col)
{
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("DenseBlock index is out of range.");
    }

    return values_[col * rows_ + row];
}

const Scalar& DenseBlock::operator()(Index row, Index col) const
{
    if (row >= rows_ || col >= cols_) {
        throw std::out_of_range("DenseBlock index is out of range.");
    }

    return values_[col * rows_ + row];
}

Scalar* DenseBlock::column(Index col)
{
    if (col >= cols_) {
        throw std::out_of_range("DenseBlock column index is out of range.");
    }

    return values_.data() + col * rows_;
}

const Scalar* DenseBlock::column(Index col) const
{
    if (col >= cols_) {
        throw std::out_of_range("DenseBlock column index is out of range.");
    }

    return values_.data() + col * rows_;
}

void DenseBlock::set_column(Index col, const Vector& values)
{
    if (col >= cols_) {
        throw std::out_of_range("DenseBlock column index is out of range.");
    }

    if (values.size() != rows_) {
        throw std::invalid_argument("DenseBlock column assignment has wrong size.");
    }

    std::copy(values.begin(), values.end(), values_.begin() + col * rows_);
}

Vector DenseBlock::get_column(Index col) const
{
    if (col >= cols_) {
        throw std::out_of_range("DenseBlock column index is out of range.");
    }

    return Vector(values_.begin() + col * rows_,
                  values_.begin() + (col + 1) * rows_);
}

DenseBlock DenseBlock::leading_columns(Index count) const
{
    if (count > cols_) {
        throw std::invalid_argument("Requested too many leading columns.");
    }

    DenseBlock result(rows_, count);

    for (Index col = 0; col < count; ++col) {
        for (Index row = 0; row < rows_; ++row) {
            result(row, col) = (*this)(row, col);
        }
    }

    return result;
}

DenseBlock DenseBlock::leading_principal_block(Index count) const
{
    if (rows_ != cols_ || count > rows_) {
        throw std::invalid_argument("Invalid leading principal block size.");
    }

    DenseBlock result(count, count);

    for (Index col = 0; col < count; ++col) {
        for (Index row = 0; row < count; ++row) {
            result(row, col) = (*this)(row, col);
        }
    }

    return result;
}

DenseBlock transpose_multiply(const DenseBlock& left,
                              const DenseBlock& right)
{
    check_same_rows(left, right);

    DenseBlock result(left.cols(), right.cols());

    if (left.rows() == 0 || left.cols() == 0 || right.cols() == 0) {
        return result;
    }

    cblas_dgemm(CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                blas_size(left.cols()),
                blas_size(right.cols()),
                blas_size(left.rows()),
                1.0,
                left.data(),
                blas_size(left.rows()),
                right.data(),
                blas_size(right.rows()),
                0.0,
                result.data(),
                blas_size(result.rows()));

    return result;
}

DenseBlock transpose_multiply(const DenseBlock& left,
                              Index left_cols,
                              const DenseBlock& right)
{
    check_same_rows(left, right);

    if (left_cols > left.cols()) {
        throw std::invalid_argument("transpose_multiply: left_cols exceeds the block width.");
    }

    DenseBlock result(left_cols, right.cols());

    if (left.rows() == 0 || left_cols == 0 || right.cols() == 0) {
        return result;
    }

    cblas_dgemm(CblasColMajor,
                CblasTrans,
                CblasNoTrans,
                blas_size(left_cols),
                blas_size(right.cols()),
                blas_size(left.rows()),
                1.0,
                left.data(),
                blas_size(left.rows()),
                right.data(),
                blas_size(right.rows()),
                0.0,
                result.data(),
                blas_size(result.rows()));

    return result;
}

DenseBlock multiply(const DenseBlock& left,
                    const DenseBlock& right)
{
    if (left.cols() != right.rows()) {
        throw std::invalid_argument("Dense block multiplication dimensions do not match.");
    }

    DenseBlock result(left.rows(), right.cols());

    if (left.rows() == 0 || left.cols() == 0 || right.cols() == 0) {
        return result;
    }

    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                blas_size(left.rows()),
                blas_size(right.cols()),
                blas_size(left.cols()),
                1.0,
                left.data(),
                blas_size(left.rows()),
                right.data(),
                blas_size(right.rows()),
                0.0,
                result.data(),
                blas_size(result.rows()));

    return result;
}

DenseBlock gram_matrix(const DenseBlock& block)
{
    DenseBlock gram(block.cols(), block.cols());

    if (block.rows() == 0 || block.cols() == 0) {
        return gram;
    }

    cblas_dsyrk(CblasColMajor,
                CblasUpper,
                CblasTrans,
                blas_size(block.cols()),
                blas_size(block.rows()),
                1.0,
                block.data(),
                blas_size(block.rows()),
                0.0,
                gram.data(),
                blas_size(gram.rows()));

    for (Index col = 0; col < gram.cols(); ++col) {
        for (Index row = col + 1; row < gram.rows(); ++row) {
            gram(row, col) = gram(col, row);
        }
    }

    return gram;
}

void subtract_product(DenseBlock& target,
                      const DenseBlock& left,
                      const DenseBlock& right)
{
    if (target.rows() != left.rows()
        || target.cols() != right.cols()
        || left.cols() != right.rows()) {
        throw std::invalid_argument("Dense block update dimensions do not match.");
    }

    if (target.rows() == 0 || target.cols() == 0 || left.cols() == 0) {
        return;
    }

    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                blas_size(target.rows()),
                blas_size(target.cols()),
                blas_size(left.cols()),
                -1.0,
                left.data(),
                blas_size(left.rows()),
                right.data(),
                blas_size(right.rows()),
                1.0,
                target.data(),
                blas_size(target.rows()));
}

void subtract_product(DenseBlock& target,
                      const DenseBlock& left,
                      Index left_cols,
                      const DenseBlock& right)
{
    if (left_cols > left.cols()) {
        throw std::invalid_argument("subtract_product: left_cols exceeds the block width.");
    }

    if (target.rows() != left.rows()
        || target.cols() != right.cols()
        || left_cols != right.rows()) {
        throw std::invalid_argument("Dense block update dimensions do not match.");
    }

    if (target.rows() == 0 || target.cols() == 0 || left_cols == 0) {
        return;
    }

    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                blas_size(target.rows()),
                blas_size(target.cols()),
                blas_size(left_cols),
                -1.0,
                left.data(),
                blas_size(left.rows()),
                right.data(),
                blas_size(right.rows()),
                1.0,
                target.data(),
                blas_size(target.rows()));
}

void multiply_add_columns(const DenseBlock& block,
                          Index cols,
                          const Vector& coefficients,
                          Vector& target)
{
    if (cols > block.cols()) {
        throw std::invalid_argument("multiply_add_columns: cols exceeds the block width.");
    }

    if (coefficients.size() != cols) {
        throw std::invalid_argument("multiply_add_columns: coefficient count does not match cols.");
    }

    if (target.size() != block.rows()) {
        throw std::invalid_argument("multiply_add_columns: target has wrong size.");
    }

    if (block.rows() == 0 || cols == 0) {
        return;
    }

    cblas_dgemv(CblasColMajor,
                CblasNoTrans,
                blas_size(block.rows()),
                blas_size(cols),
                1.0,
                block.data(),
                blas_size(block.rows()),
                coefficients.data(),
                1,
                1.0,
                target.data(),
                1);
}

void multiply_add_columns_from(const DenseBlock& block,
                               Index first_col,
                               Index cols,
                               const Vector& coefficients,
                               Vector& target)
{
    if (first_col > block.cols() || cols > block.cols() - first_col) {
        throw std::invalid_argument("multiply_add_columns_from: column range exceeds the block width.");
    }

    if (coefficients.size() != cols) {
        throw std::invalid_argument("multiply_add_columns_from: coefficient count does not match cols.");
    }

    if (target.size() != block.rows()) {
        throw std::invalid_argument("multiply_add_columns_from: target has wrong size.");
    }

    if (block.rows() == 0 || cols == 0) {
        return;
    }

    cblas_dgemv(CblasColMajor,
                CblasNoTrans,
                blas_size(block.rows()),
                blas_size(cols),
                1.0,
                block.column(first_col),
                blas_size(block.rows()),
                coefficients.data(),
                1,
                1.0,
                target.data(),
                1);
}

Vector transpose_multiply_vector(const DenseBlock& block,
                                 Index cols,
                                 const Vector& x)
{
    if (cols > block.cols()) {
        throw std::invalid_argument("transpose_multiply_vector: cols exceeds the block width.");
    }

    if (x.size() != block.rows()) {
        throw std::invalid_argument("transpose_multiply_vector: x has wrong size.");
    }

    Vector result(cols, 0.0);

    if (block.rows() == 0 || cols == 0) {
        return result;
    }

    cblas_dgemv(CblasColMajor,
                CblasTrans,
                blas_size(block.rows()),
                blas_size(cols),
                1.0,
                block.data(),
                blas_size(block.rows()),
                x.data(),
                1,
                0.0,
                result.data(),
                1);

    return result;
}

void add_in_place(DenseBlock& target,
                  const DenseBlock& source)
{
    if (target.rows() != source.rows() || target.cols() != source.cols()) {
        throw std::invalid_argument("Dense blocks have different dimensions.");
    }

    if (target.size() == 0) {
        return;
    }

    cblas_daxpy(blas_size(target.size()),
                1.0,
                source.data(),
                1,
                target.data(),
                1);
}

void right_solve_upper(DenseBlock& block,
                       const DenseBlock& upper_triangular)
{
    if (upper_triangular.rows() != upper_triangular.cols()
        || block.cols() != upper_triangular.rows()) {
        throw std::invalid_argument("Invalid dimensions for right triangular solve.");
    }

    if (block.rows() == 0 || block.cols() == 0) {
        return;
    }

    cblas_dtrsm(CblasColMajor,
                CblasRight,
                CblasUpper,
                CblasNoTrans,
                CblasNonUnit,
                blas_size(block.rows()),
                blas_size(block.cols()),
                1.0,
                upper_triangular.data(),
                blas_size(upper_triangular.rows()),
                block.data(),
                blas_size(block.rows()));
}

DenseBlock pack_columns(const VectorList& columns)
{
    if (columns.empty()) {
        return DenseBlock();
    }

    const Index rows = columns.front().size();
    DenseBlock block(rows, columns.size());

    for (Index col = 0; col < columns.size(); ++col) {
        if (columns[col].size() != rows) {
            throw std::invalid_argument("Cannot pack vectors with different sizes.");
        }

        for (Index row = 0; row < rows; ++row) {
            block(row, col) = columns[col][row];
        }
    }

    return block;
}

VectorList unpack_columns(const DenseBlock& block)
{
    VectorList columns(block.cols(), Vector(block.rows(), 0.0));

    for (Index col = 0; col < block.cols(); ++col) {
        for (Index row = 0; row < block.rows(); ++row) {
            columns[col][row] = block(row, col);
        }
    }

    return columns;
}

DenseMatrix to_dense_matrix(const DenseBlock& block)
{
    DenseMatrix result(block.rows(), Vector(block.cols(), 0.0));

    for (Index row = 0; row < block.rows(); ++row) {
        for (Index col = 0; col < block.cols(); ++col) {
            result[row][col] = block(row, col);
        }
    }

    return result;
}

}
