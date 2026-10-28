/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the matrix
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "matrix.hpp"

static void assert_matrix_eq(const my_torch::Matrix &actual, const my_torch::Matrix &expected)
{
    cr_assert_eq(actual.rows(), expected.rows());
    cr_assert_eq(actual.cols(), expected.cols());
    for (std::size_t row = 0; row < expected.rows(); row++)
        for (std::size_t col = 0; col < expected.cols(); col++)
            cr_assert_float_eq(actual(row, col), expected(row, col), 1e-9);
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
