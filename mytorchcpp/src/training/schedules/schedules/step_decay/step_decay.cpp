/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** StepDecay multiplies the rate by the factor once every few epochs; every must be at least 1
*/

#include <cmath>
#include <stdexcept>
#include <string>

#include "training/schedules/schedules/step_decay/step_decay.hpp"

namespace my_torch {

static void check_every(std::size_t every)
{
    if (every < 1)
        throw std::invalid_argument("the step must be at least 1 epoch, got " + std::to_string(every));
}

StepDecay::StepDecay(double factor, std::size_t every)
:
    Schedule("step_decay"), factor(factor), every(every)
{
    check_factor(factor);
    check_every(every);
}

double StepDecay::rate(double initial_rate, std::size_t epoch) const
{
    return initial_rate * std::pow(factor, epoch / every);
}

}
