/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the learning-rate schedules
*/

#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "optimizers/schedule.hpp"

Test(schedule, constant_keeps_the_initial_rate)
{
    my_torch::Constant schedule;

    for (std::size_t epoch : {0, 1, 10, 1000})
        cr_assert_float_eq(schedule(0.1, epoch), 0.1, 1e-15);
}

Test(schedule, constant_has_its_name)
{
    my_torch::Constant schedule;

    cr_assert_str_eq(schedule.name.c_str(), "constant");
}

Test(schedule, time_decay_divides_by_one_plus_decay_times_epoch)
{
    my_torch::TimeDecay schedule(0.5);

    cr_assert_float_eq(schedule(0.1, 0), 0.1, 1e-15);
    cr_assert_float_eq(schedule(0.1, 2), 0.05, 1e-15);
    cr_assert_float_eq(schedule(0.1, 6), 0.025, 1e-15);
}

Test(schedule, time_decay_of_zero_is_constant)
{
    cr_assert_float_eq(my_torch::TimeDecay(0.0)(0.1, 100), 0.1, 1e-15);
}

Test(schedule, time_decay_rejects_a_negative_decay)
{
    cr_assert_throw(my_torch::TimeDecay(-0.1), std::invalid_argument);
}
