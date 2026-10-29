/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Constant keeps the initial learning rate at every epoch
*/

#include "training/schedules/schedules/constant/constant.hpp"

namespace my_torch {

Constant::Constant()
:
    Schedule("constant")
{}

double Constant::rate(double initial_rate, std::size_t) const
{
    return initial_rate;
}

}
