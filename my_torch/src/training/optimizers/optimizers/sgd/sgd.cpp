/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** SGD moves each parameter against its gradient, scaled by the learning rate
*/

#include "training/optimizers/optimizers/sgd/sgd.hpp"

namespace my_torch {

SGD::SGD(double learning_rate, std::unique_ptr<Schedule> schedule)
:
    Optimizer("sgd", learning_rate, std::move(schedule))
{}

void SGD::update(Matrix &value, const Matrix &gradient)
{
    value -= gradient * learning_rate;
}

}
