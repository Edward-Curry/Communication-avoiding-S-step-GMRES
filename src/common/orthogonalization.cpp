#include "common/orthogonalization.hpp"

#include "common/vector_ops.hpp"

#include <stdexcept>

namespace gmres
{
    void modified_gram_schmidt(const std::vector<Vector>& basis,
                               Vector& w,
                               Vector& h)
    {
        h.clear();
        h.reserve(basis.size() + 1);

        for (const Vector& v : basis)
        {
            check_same_size(v, w);

            Scalar coefficient = dot(v, w);
            h.push_back(coefficient);

            axpy(-coefficient, v, w);
        }

        Scalar remaining_norm = norm2(w);
        h.push_back(remaining_norm);

        if (remaining_norm == 0.0)
        {
            throw std::runtime_error("Breakdown in modified Gram-Schmidt: zero remaining vector.");
        }

        scal(1.0 / remaining_norm, w);
    }
}