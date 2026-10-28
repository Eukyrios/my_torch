/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the dense layer
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "dense.hpp"
#include "neuron.hpp"

static const std::vector<std::vector<double>> WEIGHTS = {{0.5, -1.0}, {0.25, 0.5}};
static const std::vector<double> BIASES = {0.1, -0.3};

static std::vector<double> column(const std::vector<std::vector<double>> &weights, std::size_t j)
{
    std::vector<double> values;
    for (const std::vector<double> &row : weights)
        values.push_back(row[j]);
    return values;
}

Test(dense, forward_matches_the_hand_computation)
{
    std::vector<double> outputs = my_torch::Dense(WEIGHTS, BIASES).forward({1, 2});

    cr_assert_float_eq(outputs[0], 1.1, 1e-9);
    cr_assert_float_eq(outputs[1], -0.3, 1e-9);
}

Test(dense, each_output_is_the_neuron_of_its_column)
{
    std::vector<std::vector<double>> weights = {{0.3, -0.7, 1.2}, {-0.4, 0.9, 0.1}, {2.0, -1.5, 0.6}, {0.8, 0.2, -0.9}};
    std::vector<double> biases = {0.5, -0.2, 0.05};
    std::vector<double> inputs = {1.5, -0.5, 0.25, 2.0};
    std::vector<double> outputs = my_torch::Dense(weights, biases).forward(inputs);

    for (std::size_t j = 0; j < 3; j++)
        cr_assert_float_eq(outputs[j], my_torch::weighted_sum(inputs, column(weights, j), biases[j]), 1e-9);
}

Test(dense, sizes_come_from_the_weight_matrix)
{
    my_torch::Dense layer(std::vector<std::vector<double>>(768, std::vector<double>(256)), std::vector<double>(256));

    cr_assert_eq(layer.n_inputs(), 768u);
    cr_assert_eq(layer.n_outputs(), 256u);
}

Test(dense, rejects_weights_that_are_not_a_matrix)
{
    cr_assert_throw(my_torch::Dense({{1, 2}, {3}}, {0, 0}), std::invalid_argument);
    cr_assert_throw(my_torch::Dense({}, {}), std::invalid_argument);
}

Test(dense, rejects_a_bias_count_that_does_not_match_the_neurons)
{
    cr_assert_throw(my_torch::Dense(WEIGHTS, {0.1}), std::invalid_argument);
    cr_assert_throw(my_torch::Dense(WEIGHTS, {0.1, 0.2, 0.3}), std::invalid_argument);
}

Test(dense, rejects_the_wrong_number_of_inputs)
{
    my_torch::Dense layer(WEIGHTS, BIASES);

    cr_assert_throw(layer.forward({1}), std::invalid_argument);
    cr_assert_throw(layer.forward({1, 2, 3}), std::invalid_argument);
}
