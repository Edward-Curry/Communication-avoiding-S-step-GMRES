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

std::vector<Vector> harmonic_ritz_vectors(const DenseMatrix& hessenberg,
                                          Index leading_offset,
                                          Index num_wanted)
{
    if (hessenberg.empty() || num_wanted == 0) {
        return {};
    }

    // m = square size of the accumulated Hessenberg (its column count); the
    // full matrix is (m+1) x m, the extra row carrying the trailing
    // subdiagonal entry the harmonic correction needs.
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

    // Trailing block: the clean (m+1) x m Arnoldi Hessenberg of the (deflated)
    // operator over columns [leading_offset, total_cols). Seed columns
    // [0, leading_offset) are skipped - their Hessenberg entries are an identity
    // block, not projected-operator data.
    const Index m = total_cols - leading_offset;

    // Hbar (trailing (m+1) x m, row-major): Hbar[r][c] = H[offset+r][offset+c].
    // Row m is the trailing subdiagonal (global upper-Hessenberg structure);
    // H_sq is the leading m x m square.
    std::vector<double> hbar((m + 1) * m, 0.0);
    for (Index r = 0; r <= m; ++r) {
        for (Index c = 0; c < m; ++c) {
            hbar[r * m + c] = hessenberg[leading_offset + r][leading_offset + c];
        }
    }

    // Harmonic Ritz via the GENERALIZED eigenproblem
    //   (Hbar^T Hbar) g = theta (H_sq^T) g.
    // Its smallest-|theta| solutions are the wanted harmonic Ritz vectors. The
    // generalized form (rather than the reduced M = H_sq + h^2 H_sq^-T e e^T)
    // stays robust when H_sq is near-singular - precisely the small-eigenvalue
    // case deflation targets, where inverting H_sq overflows: dggev returns
    // those directions as beta = 0 (theta = infinity), and they are filtered.
    std::vector<double> a_gen(m * m, 0.0); // A = Hbar^T Hbar (symmetric)
    std::vector<double> b_gen(m * m, 0.0); // B = H_sq^T
    for (Index i = 0; i < m; ++i) {
        for (Index j = 0; j < m; ++j) {
            double acc = 0.0;
            for (Index r = 0; r <= m; ++r) {
                acc += hbar[r * m + i] * hbar[r * m + j];
            }
            a_gen[i * m + j] = acc;
            b_gen[i * m + j] = hbar[j * m + i]; // H_sq^T: [i][j] = H_sq[j][i]
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

    // |theta_i| = |alpha_i| / |beta_i|; beta_i == 0 means theta = infinity (a
    // direction with no small-eigenvalue content - sorted last, never chosen).
    auto magnitude = [&](Index i) {
        const Scalar alpha_mag = std::hypot(alphar[i], alphai[i]);
        return beta[i] == 0.0 ? std::numeric_limits<Scalar>::infinity()
                              : alpha_mag / std::abs(beta[i]);
    };

    // Ascending order by |theta|: smallest harmonic Ritz values first.
    std::vector<Index> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](Index a, Index b) {
        return magnitude(a) < magnitude(b);
    });

    // dggev stores a real eigenvector in column j of VR; a complex-conjugate
    // pair (j, j+1) stores the shared eigenvector's real part in column j and
    // imaginary part in column j+1. Both real vectors span the pair's 2D
    // invariant subspace, so we emit both (budget permitting) and mark the
    // partner consumed.
    std::vector<char> consumed(m, 0);
    std::vector<Vector> vectors;

    // Extracts VR column `col`; returns false if any entry is non-finite
    // (dggev can emit NaN eigenvector columns for a near-defective pencil even
    // with a finite eigenvalue - such a "vector" carries no usable direction).
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
        // Sorted ascending, so once theta is infinite (beta == 0) every
        // remaining direction is too - none carry small-eigenvalue content.
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
