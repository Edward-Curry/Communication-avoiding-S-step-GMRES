#ifndef COMMON_TYPES_HPP
#define COMMON_TYPES_HPP

#include <vector>
#include <cstddef>

namespace gmres
{
    using Scalar = double;
    using Index = std::size_t;
    using Vector = std::vector<Scalar>;
    using VectorList = std::vector<Vector>;
    using DenseMatrix = std::vector<Vector>;
}

#endif