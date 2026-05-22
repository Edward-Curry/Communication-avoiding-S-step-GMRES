#ifndef SEQUENTIAL_GMRES_SEQ_STEP_HPP
#define SEQUENTIAL_GMRES_SEQ_STEP_HPP

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

namespace gmres
{
    struct GMRESCycleResult
    {
        Vector x;
        Vector residual_history;
        Index iterations = 0;
        bool converged = false;
    };

    GMRESCycleResult gmres_seq_cycle(const SparseMatrixCSR& A,
                                     const Vector& b,
                                     const Vector& x_start,
                                     const Vector& r_start,
                                     Scalar beta,
                                     const GMRESConfig& config);
}

#endif
