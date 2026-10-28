/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the matrix
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "matrix.hpp"

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
