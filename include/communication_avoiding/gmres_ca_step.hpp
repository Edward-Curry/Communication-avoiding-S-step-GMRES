#ifndef COMMUNICATION_AVOIDING_GMRES_CA_STEP_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_STEP_HPP

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"

namespace gmres {

struct CAGMRESCycleResult {
    Vector x;
    Vector residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;
};

CAGMRESCycleResult gmres_ca_cycle(const SparseMatrixCSR& A,
                                  const Vector& b,
                                  const Vector& x_start,
                                  const Vector& r_start,
                                  Scalar beta,
                                  const GMRESConfig& config);

}

#endif