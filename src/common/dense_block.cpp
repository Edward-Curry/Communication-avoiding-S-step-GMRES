#include "common/dense_block.hpp"

#include <cblas.h>

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
