/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the neuron
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "layers/neuron.hpp"

Test(neuron, forward_matches_the_hand_computation)
{
    cr_assert_float_eq(my_torch::Neuron({0.5, 0.25}, 0.1).forward({1, 2}), 1.1, 1e-9);
}

Test(neuron, gives_the_bias_alone_when_every_input_is_zero)
{
    cr_assert_float_eq(my_torch::Neuron({3, -2, 7}, -0.5).forward({0, 0, 0}), -0.5, 1e-9);
}

Test(neuron, is_lowered_by_negative_weights)
{
    cr_assert_float_eq(my_torch::Neuron({-1, 0.5}, -0.3).forward({1, 2}), -0.3, 1e-9);
}

Test(neuron, is_called_like_a_function)
{
    my_torch::Neuron neuron({0.5, 0.25}, 0.1);

    cr_assert_float_eq(neuron({1, 2}), neuron.forward({1, 2}), 1e-12);
}

Test(neuron, has_one_input_per_weight)
{
    cr_assert_eq(my_torch::Neuron({1, 2, 3}, 0).n_inputs(), 3u);
}

Test(neuron, keeps_its_weights_and_bias)
{
    my_torch::Neuron neuron({1, 2}, 0.5);

    cr_assert(neuron.weights == std::vector<double>({1, 2}));
    cr_assert_float_eq(neuron.bias, 0.5, 1e-12);
}

Test(neuron, rejects_a_neuron_without_weights)
{
    cr_assert_throw(my_torch::Neuron({}, 0), std::invalid_argument);
}

Test(neuron, rejects_more_inputs_than_weights)
{
    cr_assert_throw(my_torch::Neuron({1, 2}, 0).forward({1, 2, 3}), std::invalid_argument);
}

Test(neuron, rejects_fewer_inputs_than_weights)
{
    cr_assert_throw(my_torch::Neuron({1, 2, 3}, 0).forward({1, 2}), std::invalid_argument);
}
