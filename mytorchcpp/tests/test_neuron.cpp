/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the neuron's weighted sum
*/

#include <criterion/criterion.h>

#include "neuron.hpp"

Test(weighted_sum, matches_the_hand_computation)
{
    cr_assert_float_eq(my_torch::weighted_sum({1, 2}, {0.5, 0.25}, 0.1), 1.1, 1e-9);
}

Test(weighted_sum, gives_the_bias_alone_when_every_input_is_zero)
{
    cr_assert_float_eq(my_torch::weighted_sum({0, 0, 0}, {3, -2, 7}, -0.5), -0.5, 1e-9);
}

Test(weighted_sum, is_lowered_by_negative_weights)
{
    cr_assert_float_eq(my_torch::weighted_sum({1, 2}, {-1, 0.5}, -0.3), -0.3, 1e-9);
}
