#include "common/sparse_matrix.hpp"

#include <cassert>
#include <print>

int main()
{
    gmres::Vector values = {
        2.0,
        3.0, 4.0,
        5.0, 6.0
    };

    std::vector<gmres::Index> col_indices = {
        0,
        1, 2,
        0, 2
    };

    std::vector<gmres::Index> row_ptr = {
        0, 1, 3, 5
    };

    gmres::SparseMatrixCSR A(3, 3, values, col_indices, row_ptr);

    gmres::Vector x = {1.0, 2.0, 3.0};

    gmres::Vector y = A.multiply(x);

    assert(y.size() == 3);
    assert(y[0] == 2.0);
    assert(y[1] == 18.0);
    assert(y[2] == 23.0);

    std::println("Sparse matrix test passed.");

    return 0;
}
