/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** ExponentialDecay multiplies the rate by the factor every epoch: initial * factor^epoch
*/

#include <cmath>

#include "training/schedules/schedules/exponential_decay/exponential_decay.hpp"

namespace my_torch {

ExponentialDecay::ExponentialDecay(double factor)
:
    Schedule("exponential_decay"), factor(factor)
{
    check_factor(factor);
}

double ExponentialDecay::rate(double initial_rate, std::size_t epoch) const
{
    return initial_rate * std::pow(factor, epoch);
}

}
