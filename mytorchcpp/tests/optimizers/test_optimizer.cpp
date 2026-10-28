/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the optimizers
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "assert_matrix.hpp"
#include "layers/dense_layer.hpp"
#include "optimizers/optimizer.hpp"

Test(sgd, moves_a_value_against_its_gradient)
{
    my_torch::Matrix value({{1, 2}, {3, 4}});

    my_torch::SGD(0.1).update(value, my_torch::Matrix({{10, -10}, {0, 5}}));
    assert_matrix_eq(value, my_torch::Matrix({{0, 3}, {3, 3.5}}));
}

Test(sgd, step_updates_every_parameter_in_place)
{
    my_torch::Matrix weights = my_torch::Matrix::row({1, 2});
    my_torch::Matrix weights_gradient = my_torch::Matrix::row({1, 1});
    my_torch::Matrix biases = my_torch::Matrix::row({0.5});
    my_torch::Matrix biases_gradient = my_torch::Matrix::row({-1});

    my_torch::SGD(0.5).step({{&weights, &weights_gradient}, {&biases, &biases_gradient}});
    assert_matrix_eq(weights, my_torch::Matrix::row({0.5, 1.5}));
    assert_matrix_eq(biases, my_torch::Matrix::row({1.0}));
}

Test(sgd, step_lowers_the_loss_of_a_layer)
{
    my_torch::DenseLayer layer(my_torch::Matrix({{0.5, -1.0}, {0.25, 0.5}}), {0.1, -0.3});
    my_torch::Matrix inputs = my_torch::Matrix::row({1, 2});
    my_torch::Matrix target = my_torch::Matrix::row({0, 0});
    my_torch::Matrix before = layer.forward(inputs);

    layer.backward(before - target);
    my_torch::SGD(0.05).step(layer.parameters());
    my_torch::Matrix after = layer.forward(inputs);
    cr_assert_lt(after.hadamard(after).sum(), before.hadamard(before).sum());
}

Test(sgd, has_its_name_and_default_learning_rate)
{
    my_torch::SGD optimizer;

    cr_assert_str_eq(optimizer.name.c_str(), "sgd");
    cr_assert_float_eq(optimizer.learning_rate, 0.01, 1e-12);
}

Test(sgd, rejects_a_learning_rate_that_is_not_positive)
{
    cr_assert_throw(my_torch::SGD(0.0), std::invalid_argument);
    cr_assert_throw(my_torch::SGD(-0.1), std::invalid_argument);
}

Test(sgd, step_needs_a_backward_first)
{
    my_torch::DenseLayer layer(my_torch::Matrix::row({0.5}), {0.1});

    cr_assert_throw(my_torch::SGD().step(layer.parameters()), std::runtime_error);
}
