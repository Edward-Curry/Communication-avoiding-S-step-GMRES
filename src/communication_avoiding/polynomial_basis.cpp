#include "communication_avoiding/polynomial_basis.hpp"

#include <algorithm>
#include <cmath>
#include <lapacke.h>
#include <limits>
#include <stdexcept>
#include <vector>

namespace gmres {

namespace {

lapack_int lapack_size(Index size)
{
    if (size > static_cast<Index>(std::numeric_limits<lapack_int>::max())) {
        throw std::length_error("polynomial_basis: matrix exceeds the LAPACK size range.");
    }

    return static_cast<lapack_int>(size);
}

}

Vector compute_ritz_shifts(const DenseMatrix& hessenberg)
{
    if (hessenberg.empty()) {
        return Vector();
    }

    // The square leading m x m block; Ritz values come from the projected
    // operator, not the rectangular (m+1) x m least-squares Hessenberg.
    const Index m = hessenberg.front().size();

    if (m == 0) {
        return Vector();
    }

    for (const Vector& row : hessenberg) {
        if (row.size() != m) {
            throw std::invalid_argument(
                "compute_ritz_shifts: hessenberg rows have inconsistent width.");
        }
    }

    if (hessenberg.size() < m) {
        throw std::invalid_argument(
            "compute_ritz_shifts: hessenberg does not have a full square leading block.");
    }

    std::vector<double> a(m * m, 0.0);

    for (Index row = 0; row < m; ++row) {
        for (Index col = 0; col < m; ++col) {
            a[row * m + col] = hessenberg[row][col];
        }
    }

    std::vector<double> wr(m, 0.0);
    std::vector<double> wi(m, 0.0);

    const lapack_int n = lapack_size(m);

    const lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR,
                                          'N',
                                          'N',
                                          n,
                                          a.data(),
                                          n,
                                          wr.data(),
                                          wi.data(),
                                          nullptr,
                                          1,
                                          nullptr,
                                          1);

    if (info != 0) {
        throw std::runtime_error("LAPACKE_dgeev failed while estimating Ritz values.");
    }

    Vector shifts;
    shifts.reserve(m);

    for (Index i = 0; i < m; ++i) {
        const Scalar magnitude = std::sqrt(wr[i] * wr[i] + wi[i] * wi[i]);
        const Scalar imaginary_tolerance =
            std::numeric_limits<Scalar>::epsilon() * 1e3 * std::max(magnitude, 1.0);

        if (std::abs(wi[i]) <= imaginary_tolerance) {
            shifts.push_back(wr[i]);
        }
    }

    return shifts;
}

Vector leja_order(const Vector& shifts)
{
    if (shifts.empty()) {
        return Vector();
    }

    std::vector<bool> used(shifts.size(), false);
    Vector ordered;
    ordered.reserve(shifts.size());

    // Seed with the largest-magnitude shift.
    Index first = 0;
    for (Index i = 1; i < shifts.size(); ++i) {
        if (std::abs(shifts[i]) > std::abs(shifts[first])) {
            first = i;
        }
    }

    ordered.push_back(shifts[first]);
    used[first] = true;

    // Greedily append whichever remaining shift maximizes the product of
    // distances to the shifts already chosen.
    for (Index chosen = 1; chosen < shifts.size(); ++chosen) {
        Index best = shifts.size();
        Scalar best_product = -1.0;

        for (Index i = 0; i < shifts.size(); ++i) {
            if (used[i]) {
                continue;
            }

            Scalar product = 1.0;
            for (Scalar picked : ordered) {
                product *= std::abs(shifts[i] - picked);
            }

            if (product > best_product) {
                best_product = product;
                best = i;
            }
        }

        ordered.push_back(shifts[best]);
        used[best] = true;
    }

    return ordered;
}

}
