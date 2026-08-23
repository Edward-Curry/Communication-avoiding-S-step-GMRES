/**
 * @file include/common/types.hpp
 * @brief Defines shared numerical type aliases.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

#include <vector>
#include <cstddef>

namespace gmres
{
    /// @brief Floating-point scalar type.
    using Scalar = double;
    /// @brief Index type for dimensions and offsets.
    using Index = std::size_t;
    /// @brief Dense vector of scalar values.
    using Vector = std::vector<Scalar>;
    /// @brief Collection of dense vectors.
    using VectorList = std::vector<Vector>;
    /// @brief Dense matrix represented as rows of vectors.
    using DenseMatrix = std::vector<Vector>;
}

#endif
