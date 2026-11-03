/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the dropout layer
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "assert_matrix.hpp"
#include "model/activations/activations/softmax/softmax.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "model/layers/layers/dropout/dropout.hpp"
#include "model/network/network.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/optimizers/optimizers/sgd/sgd.hpp"

Test(dropout, passes_its_inputs_out_of_training)
{
    my_torch::Dropout dropout(0.5);
    my_torch::Matrix inputs({{1, -2, 3}, {4, 5, -6}});

    assert_matrix_eq(dropout(inputs), inputs);
    assert_matrix_eq(dropout.backward(inputs), inputs);
}

Test(dropout, drops_about_rate_of_the_inputs_and_scales_the_others)
{
    my_torch::Dropout dropout(0.25, std::mt19937(1));
    my_torch::Matrix inputs(100, 100, 1.0);

    dropout.training = true;
    my_torch::Matrix outputs = dropout(inputs);
    std::size_t dropped = 0;
    for (std::size_t row = 0; row < 100; row++)
        for (std::size_t col = 0; col < 100; col++) {
            cr_assert(outputs(row, col) == 0.0 || outputs(row, col) == 1.0 / 0.75);
            dropped += outputs(row, col) == 0.0;
        }
    cr_assert(dropped > 2300 && dropped < 2700, "%zu of 10000 dropped", dropped);
    cr_assert_float_eq(outputs.sum() / 10000, 1.0, 0.05, "the mean of the inputs is kept");
}

Test(dropout, backward_lets_the_gradient_through_the_kept_inputs_only)
{
    my_torch::Dropout dropout(0.5, std::mt19937(3));
    my_torch::Matrix inputs(4, 6, 1.0);

    dropout.training = true;
    my_torch::Matrix outputs = dropout(inputs);
    assert_matrix_eq(dropout.backward(my_torch::Matrix(4, 6, 1.0)), outputs);
    cr_assert_throw(dropout.backward(my_torch::Matrix(2, 6, 1.0)), std::invalid_argument);
}

Test(dropout, draws_a_new_mask_each_time)
{
    my_torch::Dropout dropout(0.5, std::mt19937(5));
    my_torch::Matrix inputs(1, 50, 1.0);

    dropout.training = true;
    my_torch::Matrix first = dropout(inputs);
    my_torch::Matrix second = dropout(inputs);
    cr_assert_neq((first - second).apply([](double value) { return value * value; }).sum(), 0.0);
}

Test(dropout, a_rate_of_zero_changes_nothing)
{
    my_torch::Dropout dropout(0.0);
    my_torch::Matrix inputs({{1, 2}});

    dropout.training = true;
    assert_matrix_eq(dropout(inputs), inputs);
}

Test(dropout, has_its_name_and_no_shape_or_parameters)
{
    my_torch::Dropout dropout(0.2);

    cr_assert_str_eq(dropout.name.c_str(), "dropout");
    cr_assert_not(dropout.shape().has_value());
    cr_assert(dropout.parameters().empty());
}

Test(dropout, rejects_a_rate_outside_0_1)
{
    cr_assert_throw(my_torch::Dropout(1.0), std::invalid_argument);
    cr_assert_throw(my_torch::Dropout(-0.1), std::invalid_argument);
}

Test(dropout, is_only_active_while_the_network_trains)
{
    std::mt19937 rng(0);
    std::vector<std::unique_ptr<my_torch::Layer>> layers;
    layers.push_back(std::make_unique<my_torch::Dropout>(0.5));
    layers.push_back(my_torch::DenseLayer::create(4, 2, std::make_unique<my_torch::Softmax>(), rng));
    my_torch::Network network(std::move(layers));
    my_torch::Matrix inputs({{1, 0, 1, 0}, {0, 1, 0, 1}});
    my_torch::SGD optimizer(0.1);
    my_torch::Matrix before = network(inputs);

    cr_assert_not(network.layers[0]->training);
    assert_matrix_eq(network(inputs), before);
    network.train(inputs, my_torch::Matrix({{1, 0}, {0, 1}}), my_torch::CrossEntropy(), optimizer, 3, 2, rng);
    cr_assert_not(network.layers[0]->training);
    cr_assert_not(network.layers[1]->training);
}
