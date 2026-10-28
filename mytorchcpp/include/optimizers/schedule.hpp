/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Learning-rate schedules: the rate to use at an epoch, from the initial one; decaying it takes
** big steps early and finer ones near the end, Constant keeps it
*/

#pragma once

#include <cstddef>
#include <string>

namespace my_torch {

class Schedule {
public:
    explicit Schedule(const std::string &name);
    virtual ~Schedule() = default;

    virtual double rate(double initial_rate, std::size_t epoch) const = 0;
    double operator()(double initial_rate, std::size_t epoch) const;

    std::string name;
};

class Constant : public Schedule {
public:
    Constant();

    double rate(double initial_rate, std::size_t epoch) const override;
};

class TimeDecay : public Schedule {
public:
    explicit TimeDecay(double decay);

    double rate(double initial_rate, std::size_t epoch) const override;

    double decay;
};

class ExponentialDecay : public Schedule {
public:
    explicit ExponentialDecay(double factor);

    double rate(double initial_rate, std::size_t epoch) const override;

    double factor;
};
}
