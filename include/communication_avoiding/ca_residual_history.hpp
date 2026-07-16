#ifndef COMMUNICATION_AVOIDING_CA_RESIDUAL_HISTORY_HPP
#define COMMUNICATION_AVOIDING_CA_RESIDUAL_HISTORY_HPP

#include "common/types.hpp"

#include <vector>

namespace gmres {

struct CAResidualSample {
    Index iteration = 0;
    Scalar residual_norm = 0.0;
    // true when the residual was recomputed as ||b - A x||;
    // false when it is the Givens least-squares estimate.
    bool recomputed = false;
    // s-step block width that produced this estimate (the value chosen by the
    // adaptive-s policy for that block). 0 for the initial and restart-
    // recomputed samples, which do not correspond to a single block.
    Index block_s = 0;
};

using CAResidualHistory = std::vector<CAResidualSample>;

}

#endif
