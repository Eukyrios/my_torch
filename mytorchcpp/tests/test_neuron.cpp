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
