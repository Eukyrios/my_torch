/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Schedule is the base class of learning-rate schedules: the rate to use at an epoch, from the
** initial one; decaying it takes big steps early and finer ones near the end
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

protected:
    static void check_factor(double factor);
};

}
