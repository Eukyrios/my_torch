/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** StepDecay multiplies the rate by the factor once every few epochs:
** initial * factor^(epoch / every)
*/

#pragma once

#include <cstddef>

#include "training/schedules/schedule.hpp"

namespace my_torch {

class StepDecay : public Schedule {
public:
    StepDecay(double factor, std::size_t every);

    double rate(double initial_rate, std::size_t epoch) const override;

    double factor;
    std::size_t every;
};

}
