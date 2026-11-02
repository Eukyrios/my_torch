/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Calling a schedule gives its rate; decay factors must be in (0, 1]
*/

#include <stdexcept>
#include <string>

#include "training/schedules/schedule.hpp"

namespace my_torch {

void Schedule::check_factor(double factor)
{
    if (!(factor > 0.0 && factor <= 1.0))
        throw std::invalid_argument("the decay factor must be in (0, 1], got " + std::to_string(factor));
}

Schedule::Schedule(const std::string &name)
:
    name(name)
{}

double Schedule::operator()(double initial_rate, std::size_t epoch) const
{
    return rate(initial_rate, epoch);
}

}
