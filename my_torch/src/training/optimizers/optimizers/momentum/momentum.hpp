/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Momentum keeps a velocity per parameter, velocity = momentum * velocity - rate * gradient,
** so steps in a steady direction build up speed
*/

#pragma once

#include <memory>
#include <unordered_map>

#include "training/optimizers/optimizer.hpp"

namespace my_torch {

class Momentum : public Optimizer {
public:
    explicit Momentum(double learning_rate = 0.01, double momentum = 0.9, std::unique_ptr<Schedule> schedule = nullptr);

    void update(Matrix &value, const Matrix &gradient) override;

    double momentum;
    std::unordered_map<const Matrix *, Matrix> velocities;
};

}
