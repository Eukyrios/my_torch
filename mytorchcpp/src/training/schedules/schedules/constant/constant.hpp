/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Constant keeps the initial learning rate at every epoch
*/

#pragma once

#include "training/schedules/schedule.hpp"

namespace my_torch {

class Constant : public Schedule {
public:
    Constant();

    double rate(double initial_rate, std::size_t epoch) const override;
};

}
