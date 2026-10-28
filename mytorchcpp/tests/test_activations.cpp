/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the activation functions
*/

#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "activations.hpp"
#include "assert_matrix.hpp"

static const my_torch::Matrix Z = my_torch::Matrix::row({-2.0, -0.5, 0.0, 0.5, 2.0});

Test(activations, linear_returns_z_unchanged)
{
    assert_matrix_eq(my_torch::linear(Z), Z);
}

Test(activations, relu_cuts_negative_values_to_zero)
{
    assert_matrix_eq(my_torch::relu(Z), my_torch::Matrix::row({0.0, 0.0, 0.0, 0.5, 2.0}));
}

Test(activations, sigmoid_matches_its_definition)
{
    my_torch::Matrix outputs = my_torch::sigmoid(Z);

    for (std::size_t col = 0; col < Z.cols(); col++)
        cr_assert_float_eq(outputs(0, col), 1.0 / (1.0 + std::exp(-Z(0, col))), 1e-9);
}

Test(activations, sigmoid_is_one_half_at_zero)
{
    cr_assert_float_eq(my_torch::sigmoid(my_torch::Matrix::row({0.0}))(0, 0), 0.5, 1e-12);
}

Test(activations, sigmoid_does_not_overflow_on_large_values)
{
    assert_matrix_eq(my_torch::sigmoid(my_torch::Matrix::row({-1000.0, 1000.0})), my_torch::Matrix::row({0.0, 1.0}));
}

Test(activations, tanh_stays_between_minus_one_and_one)
{
    assert_matrix_eq(my_torch::tanh(my_torch::Matrix::row({-1000.0, 0.0, 1000.0})), my_torch::Matrix::row({-1.0, 0.0, 1.0}));
}

Test(activations, get_activation_finds_every_activation_by_name)
{
    for (const auto &[name, function] : my_torch::activations())
        cr_assert_eq(my_torch::get_activation(name), function);
    cr_assert_eq(my_torch::get_activation("relu"), my_torch::relu);
}

Test(activations, get_activation_rejects_an_unknown_name)
{
    cr_assert_throw(my_torch::get_activation("swish"), std::invalid_argument);
}
