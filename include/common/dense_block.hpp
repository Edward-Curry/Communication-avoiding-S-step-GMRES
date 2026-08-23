/**
 * @file include/common/dense_block.hpp
 * @brief Declares dense column-major block operations.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMON_DENSE_BLOCK_HPP
#define COMMON_DENSE_BLOCK_HPP

#include "common/types.hpp"

namespace gmres {

/**
 * @brief Stores a dense matrix in column-major order.
 */
class DenseBlock {
public:
    /**
     * @brief Constructs an empty block.
     */
    DenseBlock() = default;

    /**
     * @brief Constructs a block with a uniform initial value.
     * @param rows Number of rows.
     * @param cols Number of columns.
     * @param value Initial value for each entry.
     */
    DenseBlock(Index rows, Index cols, Scalar value = 0.0);

    /** @return Number of rows. */
    Index rows() const;
    /** @return Number of columns. */
    Index cols() const;
    /** @return Number of stored entries. */
    Index size() const;

    /** @return Pointer to the column-major storage. */
    Scalar* data();
    /** @return Const pointer to the column-major storage. */
    const Scalar* data() const;

    /**
     * @brief Returns a mutable matrix entry.
     * @param row Row index.
     * @param col Column index.
     * @return Reference to the selected entry.
     */
    Scalar& operator()(Index row, Index col);

    /**
     * @brief Returns a matrix entry.
     * @param row Row index.
     * @param col Column index.
     * @return Const reference to the selected entry.
     */
    const Scalar& operator()(Index row, Index col) const;

    /**
     * @brief Returns a pointer to a mutable column.
     * @param col Column index.
     * @return Pointer to the first entry of the selected column.
     */
    Scalar* column(Index col);

    /**
     * @brief Returns a pointer to a column.
     * @param col Column index.
     * @return Const pointer to the first entry of the selected column.
     */
    const Scalar* column(Index col) const;

    /**
     * @brief Replaces one block column.
     * @param col Destination column index.
     * @param values Values with length equal to rows().
     */
    void set_column(Index col, const Vector& values);

    /**
     * @brief Copies one block column.
     * @param col Source column index.
     * @return Values in the selected column.
     */
    Vector get_column(Index col) const;

    /**
     * @brief Copies the leading columns.
     * @param count Number of columns to copy.
     * @return Block containing columns zero through count minus one.
     */
    DenseBlock leading_columns(Index count) const;

    /**
     * @brief Copies the leading square principal block.
     * @param count Dimension of the square block.
     * @return Leading count by count block.
     */
    DenseBlock leading_principal_block(Index count) const;

private:
    Index rows_ = 0;
    Index cols_ = 0;
    Vector values_;
};

/**
 * @brief Computes the cross product left-transpose times right.
 * @param left Left block.
 * @param right Right block with the same row count.
 * @return Dense product.
 */
DenseBlock transpose_multiply(const DenseBlock& left,
                              const DenseBlock& right);

/**
 * @brief Computes a cross product using leading columns of the left block.
 * @param left Left block.
 * @param left_cols Number of leading columns to use.
 * @param right Right block with the same row count.
 * @return Dense product.
 */
DenseBlock transpose_multiply(const DenseBlock& left,
                              Index left_cols,
                              const DenseBlock& right);

/**
 * @brief Multiplies two dense blocks.
 * @param left Left factor.
 * @param right Right factor.
 * @return Product left times right.
 */
DenseBlock multiply(const DenseBlock& left,
                    const DenseBlock& right);

/**
 * @brief Forms a block Gram matrix.
 * @param block Input block.
 * @return block-transpose times block.
 */
DenseBlock gram_matrix(const DenseBlock& block);

/**
 * @brief Subtracts a dense product from a target block.
 * @param target Block updated in place.
 * @param left Left product factor.
 * @param right Right product factor.
 */
void subtract_product(DenseBlock& target,
                      const DenseBlock& left,
                      const DenseBlock& right);

/**
 * @brief Subtracts a product using leading columns of the left block.
 * @param target Block updated in place.
 * @param left Left product factor.
 * @param left_cols Number of leading columns to use.
 * @param right Right product factor.
 */
void subtract_product(DenseBlock& target,
                      const DenseBlock& left,
                      Index left_cols,
                      const DenseBlock& right);

/**
 * @brief Adds a weighted leading-column combination to a vector.
 * @param block Source block.
 * @param cols Number of leading columns to use.
 * @param coefficients Column coefficients.
 * @param target Vector updated in place.
 */
void multiply_add_columns(const DenseBlock& block,
                          Index cols,
                          const Vector& coefficients,
                          Vector& target);

/**
 * @brief Adds a weighted contiguous column range to a vector.
 * @param block Source block.
 * @param first_col First source column.
 * @param cols Number of columns to use.
 * @param coefficients Column coefficients.
 * @param target Vector updated in place.
 */
void multiply_add_columns_from(const DenseBlock& block,
                               Index first_col,
                               Index cols,
                               const Vector& coefficients,
                               Vector& target);

/**
 * @brief Multiplies leading block columns transposed by a vector.
 * @param block Source block.
 * @param cols Number of leading columns to use.
 * @param x Input vector.
 * @return Inner products with the selected columns.
 */
Vector transpose_multiply_vector(const DenseBlock& block,
                                 Index cols,
                                 const Vector& x);

/**
 * @brief Adds one block to another.
 * @param target Block updated in place.
 * @param source Block to add.
 */
void add_in_place(DenseBlock& target,
                  const DenseBlock& source);

/**
 * @brief Solves a right upper-triangular system in place.
 * @param block Left-hand block.
 * @param upper_triangular Upper-triangular right factor.
 */
void right_solve_upper(DenseBlock& block,
                       const DenseBlock& upper_triangular);

/**
 * @brief Packs equal-length vectors into columns of a dense block.
 * @param columns Vectors to pack.
 * @return Column-major dense block.
 */
DenseBlock pack_columns(const VectorList& columns);

/**
 * @brief Copies block columns into individual vectors.
 * @param block Source block.
 * @return Vector list containing one copy of each column.
 */
VectorList unpack_columns(const DenseBlock& block);

/**
 * @brief Converts a block to row-major nested vectors.
 * @param block Source block.
 * @return Dense matrix indexed by row then column.
 */
DenseMatrix to_dense_matrix(const DenseBlock& block);

}

#endif
