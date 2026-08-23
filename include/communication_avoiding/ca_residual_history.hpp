/**
 * @file include/communication_avoiding/ca_residual_history.hpp
 * @brief Defines CA-GMRES residual-history records.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#ifndef COMMUNICATION_AVOIDING_CA_RESIDUAL_HISTORY_HPP
#define COMMUNICATION_AVOIDING_CA_RESIDUAL_HISTORY_HPP

#include "common/types.hpp"

#include <vector>

namespace gmres {

/**
 * @brief Records one CA-GMRES residual observation.
 */
struct CAResidualSample {
    Index iteration = 0;
    Scalar residual_norm = 0.0;
    /// @brief True when residual_norm was recomputed from the current iterate.
    bool recomputed = false;
    /// @brief Accepted width for the associated block, or zero when not applicable.
    Index block_s = 0;
    /// @brief True for a residual sample after recycled-subspace seeding.
    bool from_recycle_seed = false;
};

/// @brief Ordered collection of CA-GMRES residual observations.
using CAResidualHistory = std::vector<CAResidualSample>;

}

#endif
