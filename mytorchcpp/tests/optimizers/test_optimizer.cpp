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
    cr_assert_lt((after * after).sum(), (before * before).sum());
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

Test(momentum, first_step_is_a_plain_gradient_step)
{
    my_torch::Matrix value = my_torch::Matrix::row({1, 2});

    my_torch::Momentum(0.1, 0.9).update(value, my_torch::Matrix::row({1, -1}));
    assert_matrix_eq(value, my_torch::Matrix::row({0.9, 2.1}));
}

Test(momentum, steps_build_up_speed_with_the_same_gradient)
{
    my_torch::Momentum optimizer(0.1, 0.9);
    my_torch::Matrix value = my_torch::Matrix::row({0.0});
    my_torch::Matrix gradient = my_torch::Matrix::row({1.0});

    optimizer.update(value, gradient);
    optimizer.update(value, gradient);
    cr_assert_float_eq(value(0, 0), -0.1 - 0.19, 1e-12);
}

Test(momentum, keeps_one_velocity_per_parameter)
{
    my_torch::Momentum optimizer(0.1, 0.9);
    my_torch::Matrix first = my_torch::Matrix::row({0.0});
    my_torch::Matrix second = my_torch::Matrix::row({0.0, 0.0});

    optimizer.update(first, my_torch::Matrix::row({1.0}));
    optimizer.update(second, my_torch::Matrix::row({-1.0, 2.0}));
    cr_assert_eq(optimizer.velocities.size(), 2u);
    assert_matrix_eq(optimizer.velocities.at(&second), my_torch::Matrix::row({0.1, -0.2}));
}

Test(momentum, of_zero_is_sgd)
{
    my_torch::Matrix with_momentum = my_torch::Matrix::row({1, 2});
    my_torch::Matrix with_sgd = my_torch::Matrix::row({1, 2});
    my_torch::Momentum momentum(0.1, 0.0);
    my_torch::SGD sgd(0.1);

    for (int step = 0; step < 3; step++) {
        momentum.update(with_momentum, my_torch::Matrix::row({1, -1}));
        sgd.update(with_sgd, my_torch::Matrix::row({1, -1}));
    }
    assert_matrix_eq(with_momentum, with_sgd);
}

Test(momentum, has_its_name)
{
    my_torch::Momentum optimizer;

    cr_assert_str_eq(optimizer.name.c_str(), "momentum");
}

Test(momentum, rejects_a_momentum_outside_0_and_1)
{
    cr_assert_throw(my_torch::Momentum(0.1, -0.1), std::invalid_argument);
    cr_assert_throw(my_torch::Momentum(0.1, 1.0), std::invalid_argument);
}

Test(optimizer, uses_a_constant_schedule_by_default)
{
    my_torch::SGD optimizer(0.1);

    optimizer.start_epoch(50);
    cr_assert_str_eq(optimizer.schedule->name.c_str(), "constant");
    cr_assert_float_eq(optimizer.learning_rate, 0.1, 1e-15);
}

Test(optimizer, start_epoch_follows_the_schedule)
{
    my_torch::SGD optimizer(0.1, std::make_unique<my_torch::StepDecay>(0.5, 10));

    optimizer.start_epoch(25);
    cr_assert_float_eq(optimizer.learning_rate, 0.025, 1e-15);
    cr_assert_float_eq(optimizer.initial_learning_rate, 0.1, 1e-15);
}

Test(optimizer, start_epoch_always_starts_from_the_initial_rate)
{
    my_torch::Momentum optimizer(0.1, 0.9, std::make_unique<my_torch::ExponentialDecay>(0.5));

    optimizer.start_epoch(3);
    optimizer.start_epoch(1);
    cr_assert_float_eq(optimizer.learning_rate, 0.05, 1e-15);
}

Test(optimizer, steps_use_the_scheduled_rate)
{
    my_torch::SGD optimizer(1.0, std::make_unique<my_torch::ExponentialDecay>(0.5));
    my_torch::Matrix value = my_torch::Matrix::row({0.0});

    optimizer.start_epoch(1);
    optimizer.update(value, my_torch::Matrix::row({1.0}));
    cr_assert_float_eq(value(0, 0), -0.5, 1e-15);
}
