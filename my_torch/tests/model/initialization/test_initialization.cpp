/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the weight initializations
*/

#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "model/activations/activations/relu/relu.hpp"
#include "model/activations/activations/tanh/tanh.hpp"
#include "model/initialization/initialization.hpp"

Test(initialization, each_strategy_has_its_scale)
{
    my_torch::Tanh tanh;

    cr_assert_float_eq(my_torch::initial_scale(my_torch::Initialization::He, 200, 50, tanh), std::sqrt(2.0 / 200), 1e-15);
    cr_assert_float_eq(my_torch::initial_scale(my_torch::Initialization::Xavier, 200, 50, tanh), std::sqrt(2.0 / 250), 1e-15);
    cr_assert_float_eq(my_torch::initial_scale(my_torch::Initialization::LeCun, 200, 50, tanh), std::sqrt(1.0 / 200), 1e-15);
}

Test(initialization, auto_uses_he_before_relu_and_xavier_otherwise)
{
    my_torch::ReLU relu;
    my_torch::Tanh tanh;

    cr_assert_float_eq(my_torch::initial_scale(my_torch::Initialization::Auto, 200, 50, relu), std::sqrt(2.0 / 200), 1e-15);
    cr_assert_float_eq(my_torch::initial_scale(my_torch::Initialization::Auto, 200, 50, tanh), std::sqrt(2.0 / 250), 1e-15);
}

Test(initialization, a_chosen_strategy_wins_over_the_activation)
{
    my_torch::ReLU relu;

    cr_assert_float_eq(my_torch::initial_scale(my_torch::Initialization::LeCun, 200, 50, relu), std::sqrt(1.0 / 200), 1e-15);
}

Test(initialization, names_read_back_as_the_same_strategy)
{
    for (const std::string &name : my_torch::initialization_names())
        cr_assert(my_torch::initialization_name(my_torch::initialization_from_name(name)) == name);
    cr_assert(my_torch::initialization_from_name("lecun") == my_torch::Initialization::LeCun);
}

Test(initialization, rejects_an_unknown_name)
{
    cr_assert_throw(my_torch::initialization_from_name("orthogonal"), std::invalid_argument);
}
