/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the learning-rate schedules
*/

#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "training/schedules/schedules/constant/constant.hpp"
#include "training/schedules/schedules/exponential_decay/exponential_decay.hpp"
#include "training/schedules/schedule.hpp"
#include "training/schedules/schedules/step_decay/step_decay.hpp"
#include "training/schedules/schedules/time_decay/time_decay.hpp"

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

Test(schedule, exponential_decay_multiplies_by_the_factor_each_epoch)
{
    my_torch::ExponentialDecay schedule(0.5);

    cr_assert_float_eq(schedule(0.1, 0), 0.1, 1e-15);
    cr_assert_float_eq(schedule(0.1, 1), 0.05, 1e-15);
    cr_assert_float_eq(schedule(0.1, 3), 0.0125, 1e-15);
}

Test(schedule, exponential_decay_rejects_a_factor_outside_0_and_1)
{
    cr_assert_throw(my_torch::ExponentialDecay(0.0), std::invalid_argument);
    cr_assert_throw(my_torch::ExponentialDecay(1.5), std::invalid_argument);
}

Test(schedule, step_decay_multiplies_by_the_factor_every_few_epochs)
{
    my_torch::StepDecay schedule(0.5, 10);

    cr_assert_float_eq(schedule(0.1, 0), 0.1, 1e-15);
    cr_assert_float_eq(schedule(0.1, 9), 0.1, 1e-15);
    cr_assert_float_eq(schedule(0.1, 10), 0.05, 1e-15);
    cr_assert_float_eq(schedule(0.1, 25), 0.025, 1e-15);
}

Test(schedule, step_decay_rejects_a_bad_factor_or_step)
{
    cr_assert_throw(my_torch::StepDecay(0.0, 10), std::invalid_argument);
    cr_assert_throw(my_torch::StepDecay(0.5, 0), std::invalid_argument);
}

Test(schedule, every_schedule_has_its_name)
{
    my_torch::TimeDecay time(0.1);
    my_torch::ExponentialDecay exponential(0.9);
    my_torch::StepDecay step(0.5, 10);

    cr_assert_str_eq(time.name.c_str(), "time_decay");
    cr_assert_str_eq(exponential.name.c_str(), "exponential_decay");
    cr_assert_str_eq(step.name.c_str(), "step_decay");
}
