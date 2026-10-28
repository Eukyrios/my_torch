/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Each schedule turns the initial learning rate and the epoch into the rate to use
*/

#include "optimizers/schedule.hpp"

namespace my_torch {

Schedule::Schedule(const std::string &name) : name(name)
{
}

double Schedule::operator()(double initial_rate, std::size_t epoch) const
{
    return rate(initial_rate, epoch);
}

Constant::Constant() : Schedule("constant")
{
}

double Constant::rate(double initial_rate, std::size_t) const
{
    return initial_rate;
}

}
