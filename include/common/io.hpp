#ifndef COMMON_IO_HPP
#define COMMON_IO_HPP

#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

#include <string>

namespace gmres
{
    SparseMatrixCSR read_matrix_market(const std::string& filename);

    void write_residual_history(const std::string& filename,
                                const Vector& residuals);

    void write_vector_csv(const std::string& filename,
                          const Vector& x);
}

#endif