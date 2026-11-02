/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** TimeDecay lowers the rate as initial / (1 + decay * epoch)
*/

#pragma once

#include "training/schedules/schedule.hpp"

namespace my_torch {

class TimeDecay : public Schedule {
public:
    explicit TimeDecay(double decay);

    double rate(double initial_rate, std::size_t epoch) const override;

    double decay;
};

}
