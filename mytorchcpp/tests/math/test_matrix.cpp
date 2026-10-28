/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the matrix
*/

#include <criterion/criterion.h>
#include <cmath>
#include <stdexcept>

#include "assert_matrix.hpp"
#include "math/matrix.hpp"

static double square(double value)
{
    return value * value;
}

Test(matrix, is_filled_with_one_value)
{
    my_torch::Matrix matrix(2, 3, 1.5);

    cr_assert_eq(matrix.rows(), 2u);
    cr_assert_eq(matrix.cols(), 3u);
    for (std::size_t row = 0; row < 2; row++)
        for (std::size_t col = 0; col < 3; col++)
            cr_assert_float_eq(matrix(row, col), 1.5, 1e-12);
}

Test(matrix, is_built_from_its_rows)
{
    my_torch::Matrix matrix({{1, 2, 3}, {4, 5, 6}});

    cr_assert_eq(matrix.rows(), 2u);
    cr_assert_eq(matrix.cols(), 3u);
    cr_assert_float_eq(matrix(0, 2), 3.0, 1e-12);
    cr_assert_float_eq(matrix(1, 0), 4.0, 1e-12);
}

Test(matrix, has_writable_values)
{
    my_torch::Matrix matrix(2, 2);

    matrix(1, 0) = 7.0;
    cr_assert_float_eq(matrix(1, 0), 7.0, 1e-12);
    cr_assert_float_eq(matrix(0, 1), 0.0, 1e-12);
}

Test(matrix, is_empty_without_rows)
{
    my_torch::Matrix matrix(std::vector<std::vector<double>>{});

    cr_assert_eq(matrix.rows(), 0u);
    cr_assert_eq(matrix.cols(), 0u);
}

Test(matrix, row_is_one_row_of_values)
{
    assert_matrix_eq(my_torch::Matrix::row({1, 2, 3}), my_torch::Matrix({{1, 2, 3}}));
}

Test(matrix, rejects_rows_of_different_sizes)
{
    cr_assert_throw(my_torch::Matrix({{1, 2}, {3}}), std::invalid_argument);
}

Test(matrix, product_matches_the_hand_computation)
{
    my_torch::Matrix left({{1, 2}, {3, 4}, {5, 6}});
    my_torch::Matrix right({{1, 0, -1}, {2, 1, 0}});

    assert_matrix_eq(left * right, my_torch::Matrix({{5, 2, -1}, {11, 4, -3}, {17, 6, -5}}));
}

Test(matrix, product_with_the_identity_keeps_the_matrix)
{
    my_torch::Matrix matrix({{1.5, -2}, {0.25, 3}});
    my_torch::Matrix identity({{1, 0}, {0, 1}});

    assert_matrix_eq(matrix * identity, matrix);
    assert_matrix_eq(identity * matrix, matrix);
}

Test(matrix, product_rejects_sizes_that_do_not_chain)
{
    cr_assert_throw(my_torch::Matrix(2, 3) * my_torch::Matrix(2, 3), std::invalid_argument);
}

Test(matrix, transpose_swaps_rows_and_columns)
{
    my_torch::Matrix matrix({{1, 2, 3}, {4, 5, 6}});

    assert_matrix_eq(matrix.transpose(), my_torch::Matrix({{1, 4}, {2, 5}, {3, 6}}));
}

Test(matrix, transpose_twice_gives_the_matrix_back)
{
    my_torch::Matrix matrix({{1.5, -2}, {0.25, 3}, {7, 8}});

    assert_matrix_eq(matrix.transpose().transpose(), matrix);
}

Test(matrix, adds_element_by_element)
{
    my_torch::Matrix left({{1, 2}, {3, 4}});
    my_torch::Matrix right({{0.5, -2}, {10, 0}});

    assert_matrix_eq(left + right, my_torch::Matrix({{1.5, 0}, {13, 4}}));
}

Test(matrix, subtracts_element_by_element)
{
    my_torch::Matrix left({{1, 2}, {3, 4}});
    my_torch::Matrix right({{0.5, -2}, {10, 0}});

    assert_matrix_eq(left - right, my_torch::Matrix({{0.5, 4}, {-7, 4}}));
}

Test(matrix, adds_and_subtracts_in_place)
{
    my_torch::Matrix matrix({{1, 2}, {3, 4}});

    matrix += my_torch::Matrix({{1, 1}, {1, 1}});
    matrix -= my_torch::Matrix({{0, 2}, {4, 0}});
    assert_matrix_eq(matrix, my_torch::Matrix({{2, 1}, {0, 5}}));
}

Test(matrix, element_wise_operations_reject_different_shapes)
{
    cr_assert_throw(my_torch::Matrix(2, 3) + my_torch::Matrix(3, 2), std::invalid_argument);
    cr_assert_throw(my_torch::Matrix(2, 3) - my_torch::Matrix(2, 2), std::invalid_argument);
    cr_assert_throw(my_torch::Matrix(2, 3) + my_torch::Matrix::row({1, 2}), std::invalid_argument);
    cr_assert_throw(my_torch::Matrix(2, 3) + my_torch::Matrix(3, 1), std::invalid_argument);
}

Test(matrix, repeats_a_single_row_on_every_row)
{
    my_torch::Matrix matrix({{1, 2}, {3, 4}, {5, 6}});
    my_torch::Matrix row = my_torch::Matrix::row({10, -1});

    assert_matrix_eq(matrix + row, my_torch::Matrix({{11, 1}, {13, 3}, {15, 5}}));
    assert_matrix_eq(matrix - row, my_torch::Matrix({{-9, 3}, {-7, 5}, {-5, 7}}));
}

Test(matrix, is_scaled_by_a_number_on_either_side)
{
    my_torch::Matrix matrix({{1, -2}, {0.5, 4}});
    my_torch::Matrix expected({{3, -6}, {1.5, 12}});

    assert_matrix_eq(matrix * 3.0, expected);
    assert_matrix_eq(3.0 * matrix, expected);
}

Test(matrix, is_divided_by_a_number)
{
    assert_matrix_eq(my_torch::Matrix({{1, -2}, {0.5, 4}}) / 2.0, my_torch::Matrix({{0.5, -1}, {0.25, 2}}));
}

Test(matrix, is_scaled_in_place)
{
    my_torch::Matrix matrix({{1, -2}, {0.5, 4}});

    matrix *= 4.0;
    matrix /= 2.0;
    assert_matrix_eq(matrix, my_torch::Matrix({{2, -4}, {1, 8}}));
}

Test(matrix, applies_a_function_to_every_value)
{
    assert_matrix_eq(my_torch::Matrix({{1, -2}, {3, 0.5}}).apply(square), my_torch::Matrix({{1, 4}, {9, 0.25}}));
}

Test(matrix, applies_a_function_from_cmath)
{
    my_torch::Matrix result = my_torch::Matrix({{0, 1}}).apply(std::exp);

    cr_assert_float_eq(result(0, 0), 1.0, 1e-12);
    cr_assert_float_eq(result(0, 1), std::exp(1.0), 1e-12);
}

Test(matrix, row_sums_give_one_sum_per_row_as_a_column)
{
    assert_matrix_eq(my_torch::Matrix({{1, 2, 3}, {4, 5, 6}}).row_sums(), my_torch::Matrix({{6}, {15}}));
}

Test(matrix, column_sums_give_one_sum_per_column_as_a_row)
{
    assert_matrix_eq(my_torch::Matrix({{1, 2, 3}, {4, 5, 6}}).column_sums(), my_torch::Matrix::row({5, 7, 9}));
}

Test(matrix, row_maxima_give_the_largest_value_of_each_row)
{
    assert_matrix_eq(my_torch::Matrix({{1, 7, 3}, {-4, -5, -6}}).row_maxima(), my_torch::Matrix({{7}, {-4}}));
}

Test(matrix, repeats_a_single_column_on_every_column)
{
    my_torch::Matrix matrix({{1, 2}, {3, 4}, {5, 6}});
    my_torch::Matrix column({{10}, {20}, {30}});

    assert_matrix_eq(matrix + column, my_torch::Matrix({{11, 12}, {23, 24}, {35, 36}}));
    assert_matrix_eq(matrix - column, my_torch::Matrix({{-9, -8}, {-17, -16}, {-25, -24}}));
}

Test(matrix, divides_element_by_element)
{
    assert_matrix_eq(my_torch::Matrix({{1, 9}, {4, -6}}) / my_torch::Matrix({{2, 3}, {4, 2}}), my_torch::Matrix({{0.5, 3}, {1, -3}}));
}

Test(matrix, divides_each_row_by_its_own_value)
{
    my_torch::Matrix matrix({{2, 4}, {9, 3}});

    assert_matrix_eq(matrix / my_torch::Matrix({{2}, {3}}), my_torch::Matrix({{1, 2}, {3, 1}}));
}

Test(matrix, size_counts_every_value)
{
    cr_assert_eq(my_torch::Matrix(3, 4).size(), 12u);
}

Test(matrix, sum_adds_every_value)
{
    cr_assert_float_eq(my_torch::Matrix({{1, 2, 3}, {4, 5, -6}}).sum(), 9.0, 1e-12);
}

Test(matrix, hadamard_multiplies_element_by_element)
{
    my_torch::Matrix left({{1, 2}, {3, 4}});

    assert_matrix_eq(left.hadamard(my_torch::Matrix({{2, 0}, {-1, 0.5}})), my_torch::Matrix({{2, 0}, {-3, 2}}));
}

Test(matrix, hadamard_broadcasts_a_single_row)
{
    my_torch::Matrix left({{1, 2}, {3, 4}});

    assert_matrix_eq(left.hadamard(my_torch::Matrix::row({10, -1})), my_torch::Matrix({{10, -2}, {30, -4}}));
}

Test(matrix, clip_keeps_every_value_between_two_bounds)
{
    assert_matrix_eq(my_torch::Matrix({{-5, 0.5}, {2, 1}}).clip(0.0, 1.0), my_torch::Matrix({{0, 0.5}, {1, 1}}));
}
