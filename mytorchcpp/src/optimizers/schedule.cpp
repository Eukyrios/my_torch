/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Each schedule turns the initial learning rate and the epoch into the rate to use
*/

#include <cmath>
#include <stdexcept>
#include <string>

#include "optimizers/schedule.hpp"

namespace my_torch {

static void check_factor(double factor)
{
    if (!(factor > 0.0 && factor <= 1.0))
        throw std::invalid_argument("the decay factor must be in (0, 1], got " + std::to_string(factor));
}

static void check_every(std::size_t every)
{
    if (every < 1)
        throw std::invalid_argument("the step must be at least 1 epoch, got " + std::to_string(every));
}

static void check_decay(double decay)
{
    if (decay < 0.0)
        throw std::invalid_argument("the decay must not be negative, got " + std::to_string(decay));
}

Schedule::Schedule(const std::string &name)
:
    name(name)
{}

double Schedule::operator()(double initial_rate, std::size_t epoch) const
{
    return rate(initial_rate, epoch);
}

Constant::Constant()
:
    Schedule("constant")
{}

double Constant::rate(double initial_rate, std::size_t) const
{
    return initial_rate;
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
