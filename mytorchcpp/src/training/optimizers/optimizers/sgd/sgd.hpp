/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** SGD, stochastic gradient descent: each parameter moves against its gradient,
** value -= learning_rate * gradient
*/

#pragma once

#include <memory>

#include "training/optimizers/optimizer.hpp"

namespace my_torch {

class SGD : public Optimizer {
public:
    explicit SGD(double learning_rate = 0.01, std::unique_ptr<Schedule> schedule = nullptr);

    void update(Matrix &value, const Matrix &gradient) override;
};

}
