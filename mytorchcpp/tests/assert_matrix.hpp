/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Compares two matrices value by value in the tests
*/

#pragma once

#include <criterion/criterion.h>

#include "matrix.hpp"

inline void assert_matrix_eq(const my_torch::Matrix &actual, const my_torch::Matrix &expected)
{
    cr_assert_eq(actual.rows(), expected.rows());
    cr_assert_eq(actual.cols(), expected.cols());
    for (std::size_t row = 0; row < expected.rows(); row++)
        for (std::size_t col = 0; col < expected.cols(); col++)
            cr_assert_float_eq(actual(row, col), expected(row, col), 1e-9);
}
