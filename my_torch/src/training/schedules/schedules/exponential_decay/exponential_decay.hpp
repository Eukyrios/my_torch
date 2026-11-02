/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** ExponentialDecay multiplies the rate by the factor every epoch: initial * factor^epoch
*/

#pragma once

#include "training/schedules/schedule.hpp"

namespace my_torch {

class ExponentialDecay : public Schedule {
public:
    explicit ExponentialDecay(double factor);

    double rate(double initial_rate, std::size_t epoch) const override;

    double factor;
};

}
