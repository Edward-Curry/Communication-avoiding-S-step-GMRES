/**
 * @file include/communication_avoiding/gmres_ca.hpp
 * @brief Declares sequential communication-avoiding GMRES.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_GMRES_CA_HPP
#define COMMUNICATION_AVOIDING_GMRES_CA_HPP

#include "common/config.hpp"
#include "common/sparse_matrix.hpp"
#include "common/types.hpp"
#include "communication_avoiding/ca_residual_history.hpp"

namespace gmres {

/**
 * @brief Stores the result of a sequential CA-GMRES solve.
 */
struct CAGMRESResult {
    Vector x;
    CAResidualHistory residual_history;
    Index blocks_completed = 0;
    Index iterations = 0;
    bool converged = false;
};

/**
 * @brief Solves a sparse linear system with restarted CA-GMRES.
 * @param A System matrix.
 * @param b Right-hand side.
 * @param x0 Initial solution estimate.
 * @param config Solver configuration.
 * @return Final solution, residual history, iteration count, and status.
 */
CAGMRESResult gmres_ca(const SparseMatrixCSR& A,
                       const Vector& b,
                       const Vector& x0,
                       const GMRESConfig& config);

}

#endif
