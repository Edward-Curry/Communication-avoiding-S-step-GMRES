#include "communication_avoiding/hessenberg_assembly.hpp"

#include <stdexcept>

namespace gmres {

namespace {

void check_dimensions(const DenseMatrix& hessenberg,
                      const DenseMatrix& r_old,
                      const DenseMatrix& r_block)
{
    if (r_old.empty()) {
        throw std::invalid_argument(
            "Hessenberg assembly requires a nonempty old basis.");
    }

    const Index old_size = r_old.size();
    const Index block_size = r_block.size();
    const Index previous_columns = old_size - 1;

    if (block_size == 0) {
        throw std::invalid_argument(
            "Hessenberg assembly requires a nonempty block.");
    }

    for (const Vector& row : r_old) {
        if (row.size() != block_size) {
            throw std::invalid_argument(
                "R_old dimensions do not match the block size.");
        }
    }

    for (const Vector& row : r_block) {
        if (row.size() != block_size) {
            throw std::invalid_argument("R_block must be square.");
        }
    }

    if (hessenberg.size() != old_size) {
        throw std::invalid_argument(
            "Hessenberg row count does not match the old basis.");
    }

    for (const Vector& row : hessenberg) {
        if (row.size() != previous_columns) {
            throw std::invalid_argument(
                "Hessenberg column count does not match the old basis.");
        }
    }
}

}

void append_monomial_hessenberg_block(DenseMatrix& hessenberg,
                                       const DenseMatrix& r_old,
                                       const DenseMatrix& r_block)
{
    check_dimensions(hessenberg, r_old, r_block);

    const Index old_size = r_old.size();
    const Index block_size = r_block.size();
    const Index previous_columns = old_size - 1;
    const Index new_rows = old_size + block_size;
    const Index new_columns = previous_columns + block_size;

    DenseMatrix right_hand_side(new_rows, Vector(block_size, 0.0));

    for (Index row = 0; row < old_size; ++row) {
        for (Index col = 0; col < block_size; ++col) {
            right_hand_side[row][col] = r_old[row][col];
        }
    }

    for (Index row = 0; row < block_size; ++row) {
        for (Index col = 0; col < block_size; ++col) {
            right_hand_side[old_size + row][col] = r_block[row][col];
        }
    }

    // A K_left = K_right. Remove contributions from previously assembled
    // Arnoldi columns before solving for the new Hessenberg block.
    for (Index col = 1; col < block_size; ++col) {
        for (Index old_col = 0; old_col < previous_columns; ++old_col) {
            const Scalar coefficient = r_old[old_col][col - 1];

            for (Index row = 0; row < old_size; ++row) {
                right_hand_side[row][col] -=
                    hessenberg[row][old_col] * coefficient;
            }
        }
    }

    DenseMatrix change_of_coordinates(block_size,
                                      Vector(block_size, 0.0));
    change_of_coordinates[0][0] = 1.0;

    for (Index col = 1; col < block_size; ++col) {
        change_of_coordinates[0][col] =
            r_old[old_size - 1][col - 1];
    }

    for (Index row = 1; row < block_size; ++row) {
        for (Index col = row; col < block_size; ++col) {
            change_of_coordinates[row][col] =
                r_block[row - 1][col - 1];
        }
    }

    // Solve H_block * change_of_coordinates = right_hand_side.
    for (Index row = 0; row < new_rows; ++row) {
        for (Index col = 0; col < block_size; ++col) {
            Scalar value = right_hand_side[row][col];

            for (Index k = 0; k < col; ++k) {
                value -= right_hand_side[row][k] *
                         change_of_coordinates[k][col];
            }

            const Scalar diagonal = change_of_coordinates[col][col];

            if (diagonal == 0.0) {
                throw std::runtime_error(
                    "Hessenberg assembly encountered a singular basis transformation.");
            }

            right_hand_side[row][col] = value / diagonal;
        }
    }

    for (Vector& row : hessenberg) {
        row.resize(new_columns, 0.0);
    }
    hessenberg.resize(new_rows, Vector(new_columns, 0.0));

    for (Index col = 0; col < block_size; ++col) {
        const Index global_col = previous_columns + col;

        for (Index row = 0; row < new_rows; ++row) {
            hessenberg[row][global_col] =
                row <= global_col + 1 ? right_hand_side[row][col] : 0.0;
        }
    }
}

void append_shifted_hessenberg_block(DenseMatrix& hessenberg,
                                     const DenseMatrix& r_old,
                                     const DenseMatrix& r_block,
                                     const Vector& shifts,
                                     const Vector& scales)
{
    check_dimensions(hessenberg, r_old, r_block);

    const Index old_size = r_old.size();
    const Index block_size = r_block.size();
    const Index previous_columns = old_size - 1;
    const Index new_rows = old_size + block_size;
    const Index new_columns = previous_columns + block_size;

    if (shifts.size() != block_size) {
        throw std::invalid_argument(
            "append_shifted_hessenberg_block: shifts size must match the block size.");
    }

    if (!scales.empty() && scales.size() != block_size) {
        throw std::invalid_argument(
            "append_shifted_hessenberg_block: scales size must match the block size when non-empty.");
    }

    auto scale_of = [&](Index col) {
        return scales.empty() ? 1.0 : scales[col];
    };

    DenseMatrix right_hand_side(new_rows, Vector(block_size, 0.0));

    // Base term: scale[col] * w_col's own decomposition (w_col = the block's
    // own column, already known via BCGS2-CholQR's r_old/r_block).
    for (Index row = 0; row < old_size; ++row) {
        for (Index col = 0; col < block_size; ++col) {
            right_hand_side[row][col] = scale_of(col) * r_old[row][col];
        }
    }

    for (Index row = 0; row < block_size; ++row) {
        for (Index col = 0; col < block_size; ++col) {
            right_hand_side[old_size + row][col] = scale_of(col) * r_block[row][col];
        }
    }

    // Shift term: A*w_{col-1} = scale[col]*w_col + shifts[col]*w_{col-1}, so
    // the right-hand side also needs + shifts[col] * w_{col-1}'s own
    // decomposition. w_{-1} := q_prev (old_basis's last column), whose
    // decomposition is trivially the unit vector at row previous_columns.
    if (shifts[0] != 0.0) {
        right_hand_side[previous_columns][0] += shifts[0];
    }

    for (Index col = 1; col < block_size; ++col) {
        if (shifts[col] == 0.0) {
            continue;
        }

        for (Index row = 0; row < old_size; ++row) {
            right_hand_side[row][col] += shifts[col] * r_old[row][col - 1];
        }

        for (Index row = 0; row < block_size; ++row) {
            right_hand_side[old_size + row][col] += shifts[col] * r_block[row][col - 1];
        }
    }

    // Everything below is unchanged from append_monomial_hessenberg_block:
    // removing already-known old-basis contributions, building the
    // change-of-coordinates matrix, and the forward-substitution solve do
    // not depend on the recurrence, only on r_old/r_block themselves.
    for (Index col = 1; col < block_size; ++col) {
        for (Index old_col = 0; old_col < previous_columns; ++old_col) {
            const Scalar coefficient = r_old[old_col][col - 1];

            for (Index row = 0; row < old_size; ++row) {
                right_hand_side[row][col] -=
                    hessenberg[row][old_col] * coefficient;
            }
        }
    }

    DenseMatrix change_of_coordinates(block_size,
                                      Vector(block_size, 0.0));
    change_of_coordinates[0][0] = 1.0;

    for (Index col = 1; col < block_size; ++col) {
        change_of_coordinates[0][col] =
            r_old[old_size - 1][col - 1];
    }

    for (Index row = 1; row < block_size; ++row) {
        for (Index col = row; col < block_size; ++col) {
            change_of_coordinates[row][col] =
                r_block[row - 1][col - 1];
        }
    }

    for (Index row = 0; row < new_rows; ++row) {
        for (Index col = 0; col < block_size; ++col) {
            Scalar value = right_hand_side[row][col];

            for (Index k = 0; k < col; ++k) {
                value -= right_hand_side[row][k] *
                         change_of_coordinates[k][col];
            }

            const Scalar diagonal = change_of_coordinates[col][col];

            if (diagonal == 0.0) {
                throw std::runtime_error(
                    "Hessenberg assembly encountered a singular basis transformation.");
            }

            right_hand_side[row][col] = value / diagonal;
        }
    }

    for (Vector& row : hessenberg) {
        row.resize(new_columns, 0.0);
    }
    hessenberg.resize(new_rows, Vector(new_columns, 0.0));

    for (Index col = 0; col < block_size; ++col) {
        const Index global_col = previous_columns + col;

        for (Index row = 0; row < new_rows; ++row) {
            hessenberg[row][global_col] =
                row <= global_col + 1 ? right_hand_side[row][col] : 0.0;
        }
    }
}

}
