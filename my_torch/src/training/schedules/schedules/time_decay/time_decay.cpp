/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** TimeDecay lowers the rate as initial / (1 + decay * epoch); the decay must not be negative
*/

#include <stdexcept>
#include <string>

#include "training/schedules/schedules/time_decay/time_decay.hpp"

namespace my_torch {

static void check_decay(double decay)
{
    if (decay < 0.0)
        throw std::invalid_argument("the decay must not be negative, got " + std::to_string(decay));
}

TimeDecay::TimeDecay(double decay)
:
    Schedule("time_decay"), decay(decay)
{
    check_decay(decay);
}

double TimeDecay::rate(double initial_rate, std::size_t epoch) const
{
    return initial_rate / (1.0 + decay * epoch);
}

}
