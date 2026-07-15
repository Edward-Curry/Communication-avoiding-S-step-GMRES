#ifndef COMMUNICATION_AVOIDING_GMRES_CA_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_HPP

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/ca_residual_history.hpp"

namespace gmres {

struct CAGMRESResult {
    Vector x;
    CAResidualHistory residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;
};

CAGMRESResult gmres_ca(const SparseMatrixCSR& A,
                       const Vector& b,
                       const Vector& x0,
                       const GMRESConfig& config);

}

#endif