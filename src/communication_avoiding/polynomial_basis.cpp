/**
 * @file src/communication_avoiding/polynomial_basis.cpp
 * @brief Implements Ritz shifts and harmonic-Ritz recycling vectors.
 * @author Edward Curry
 * @date 2026-08-23
 * @details Last updated by Edward Curry on 2026-08-23.
 */

#include "communication_avoiding/polynomial_basis.hpp"

#include <algorithm>
#include <cmath>
#include <lapacke.h>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace gmres {

namespace {

/**
 * @brief Converts a project dimension to the LAPACK integer type.
 * @param size Dimension to convert.
 * @return LAPACK-compatible dimension.
 */
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

    // Ritz values come from the square projected Hessenberg block.
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

    // Column-major LAPACK sees the transpose, which has the same eigenvalues.
    const lapack_int info = LAPACKE_dgeev(LAPACK_COL_MAJOR,
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

    // Start from the shift with greatest magnitude.
    Index first = 0;
    for (Index i = 1; i < shifts.size(); ++i) {
        if (std::abs(shifts[i]) > std::abs(shifts[first])) {
            first = i;
        }
    }

    ordered.push_back(shifts[first]);
    used[first] = true;

    // Select the remaining shift farthest from those already chosen.
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

std::vector<Vector> harmonic_ritz_vectors(const DenseMatrix& hessenberg,
                                          Index leading_offset,
                                          Index num_wanted)
{
    if (hessenberg.empty() || num_wanted == 0) {
        return {};
    }

    // The trailing row contains the Arnoldi subdiagonal contribution.
    const Index total_cols = hessenberg.front().size();

    if (leading_offset >= total_cols) {
        return {};
    }

    for (const Vector& row : hessenberg) {
        if (row.size() != total_cols) {
            throw std::invalid_argument(
                "harmonic_ritz_vectors: hessenberg rows have inconsistent width.");
        }
    }

    if (hessenberg.size() < total_cols + 1) {
        throw std::invalid_argument(
            "harmonic_ritz_vectors: hessenberg lacks its (m+1)-th subdiagonal row.");
    }

    // Skip leading recycled columns when forming the projected operator.
    const Index m = total_cols - leading_offset;

    // Extract the trailing Arnoldi Hessenberg block.
    std::vector<double> hbar((m + 1) * m, 0.0);
    for (Index r = 0; r <= m; ++r) {
        for (Index c = 0; c < m; ++c) {
            hbar[r * m + c] = hessenberg[leading_offset + r][leading_offset + c];
        }
    }

    // Solve the generalized harmonic-Ritz eigenproblem without inverting H_sq.
    std::vector<double> a_gen(m * m, 0.0);
    std::vector<double> b_gen(m * m, 0.0);
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < m; ++j) {
            double acc = 0.0;
            for (Index r = 0; r <= m; ++r) {
                acc += hbar[r * m + i] * hbar[r * m + j];
            }
            a_gen[i * m + j] = acc;
            b_gen[i * m + j] = hbar[j * m + i];
        }
    }

    const lapack_int n = lapack_size(m);

    std::vector<double> alphar(m, 0.0);
    std::vector<double> alphai(m, 0.0);
    std::vector<double> beta(m, 0.0);
    std::vector<double> vr(m * m, 0.0);

    const lapack_int info = LAPACKE_dggev(LAPACK_ROW_MAJOR,
                                          'N',
                                          'V',
                                          n,
                                          a_gen.data(),
                                          n,
                                          b_gen.data(),
                                          n,
                                          alphar.data(),
                                          alphai.data(),
                                          beta.data(),
                                          nullptr,
                                          1,
                                          vr.data(),
                                          n);

    if (info != 0) {
        throw std::runtime_error(
            "LAPACKE_dggev failed while computing harmonic Ritz vectors.");
    }

    // Infinite generalized eigenvalues are sorted last.
    auto magnitude = [&](Index i) {
        const Scalar alpha_mag = std::hypot(alphar[i], alphai[i]);
        return beta[i] == 0.0 ? std::numeric_limits<Scalar>::infinity()
                              : alpha_mag / std::abs(beta[i]);
    };

    // Retain directions associated with the smallest harmonic Ritz values.
    std::vector<Index> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](Index a, Index b) {
        return magnitude(a) < magnitude(b);
    });

    // Complex pairs contribute their real and imaginary columns.
    std::vector<char> consumed(m, 0);
    std::vector<Vector> vectors;

    // Reject non-finite eigenvector columns.
    auto finite_column = [&](Index col, Vector& out) {
        out.assign(m, 0.0);
        for (Index i = 0; i < m; ++i) {
            const Scalar value = vr[i * m + col];
            if (!std::isfinite(value)) {
                return false;
            }
            out[i] = value;
        }
        return true;
    };

    for (Index idx : order) {
        if (static_cast<Index>(vectors.size()) >= num_wanted) {
            break;
        }
        // Remaining infinite eigenvalues cannot provide retained directions.
        if (!std::isfinite(magnitude(idx))) {
            break;
        }
        if (consumed[idx]) {
            continue;
        }

        if (alphai[idx] == 0.0) {
            Vector g;
            if (finite_column(idx, g)) {
                vectors.push_back(std::move(g));
            }
            consumed[idx] = 1;
        } else {
            const Index partner = (alphai[idx] > 0.0) ? idx + 1 : idx - 1;
            const Index real_col = (alphai[idx] > 0.0) ? idx : partner;
            const Index imag_col = (alphai[idx] > 0.0) ? partner : idx;

            consumed[idx] = 1;
            if (partner >= 0 && partner < m) {
                consumed[partner] = 1;
            }

            Vector g_real;
            if (finite_column(real_col, g_real)) {
                vectors.push_back(std::move(g_real));
            }

            Vector g_imag;
            if (static_cast<Index>(vectors.size()) < num_wanted
                && finite_column(imag_col, g_imag)) {
                vectors.push_back(std::move(g_imag));
            }
        }
    }

    return vectors;
}

}
