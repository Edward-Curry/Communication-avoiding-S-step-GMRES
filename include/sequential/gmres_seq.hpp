#ifndef SEQUENTIAL_GMRES_SEQ_HPP
#define SEQUENTIAL_GMRES_SEQ_HPP

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

namespace gmres
{
    struct GMRESResult
    {
        Vector x;
        Vector residual_history;
        Index iterations = 0;
        bool converged = false;
    };

    GMRESResult gmres_seq(const SparseMatrixCSR& A,
                          const Vector& b,
                          const Vector& x0,
                          const GMRESConfig& config);
}

#endif