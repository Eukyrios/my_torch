/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Optimizer is the base class: a name, a learning rate following a schedule (start_epoch) and
** step(parameters) after each backward pass, which calls update on every parameter
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "model/layers/layer.hpp"
#include "math/matrix.hpp"
#include "training/schedules/schedule.hpp"

namespace my_torch {

class Optimizer {
public:
    Optimizer(const std::string &name, double learning_rate, std::unique_ptr<Schedule> schedule = nullptr);
    virtual ~Optimizer() = default;

    void start_epoch(std::size_t epoch);
    void step(const std::vector<Parameter> &parameters);
    virtual void update(Matrix &value, const Matrix &gradient) = 0;

    std::string name;
    double initial_learning_rate;
    double learning_rate;
    std::unique_ptr<Schedule> schedule;
};

}
