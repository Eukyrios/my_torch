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
