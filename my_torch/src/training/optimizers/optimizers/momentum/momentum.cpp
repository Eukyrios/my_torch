/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Momentum keeps one velocity per parameter, starting at zero; momentum must be in [0, 1)
*/

#include <stdexcept>
#include <string>

#include "training/optimizers/optimizers/momentum/momentum.hpp"

namespace my_torch {

static void check_momentum(double momentum)
{
    if (!(momentum >= 0.0 && momentum < 1.0))
        throw std::invalid_argument("the momentum must be in [0, 1), got " + std::to_string(momentum));
}

Momentum::Momentum(double learning_rate, double momentum, std::unique_ptr<Schedule> schedule)
:
    Optimizer("momentum", learning_rate, std::move(schedule)), momentum(momentum)
{
    check_momentum(momentum);
}

void Momentum::update(Matrix &value, const Matrix &gradient)
{
    auto found = velocities.find(&value);
    Matrix velocity = found == velocities.end() ? Matrix(value.rows(), value.cols()) : found->second;
    velocity = velocity * momentum - gradient * learning_rate;
    velocities[&value] = velocity;
    value += velocity;
}

}
