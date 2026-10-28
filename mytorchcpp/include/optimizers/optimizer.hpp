/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Optimizer is the base class: a name, a learning rate following a schedule (start_epoch) and
** step(parameters) after each backward pass; SGD steps against the gradient, Momentum keeps a velocity
*/

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "layers/layer.hpp"
#include "math/matrix.hpp"
#include "optimizers/schedule.hpp"

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

class SGD : public Optimizer {
public:
    explicit SGD(double learning_rate = 0.01, std::unique_ptr<Schedule> schedule = nullptr);

    void update(Matrix &value, const Matrix &gradient) override;
};

class Momentum : public Optimizer {
public:
    explicit Momentum(double learning_rate = 0.01, double momentum = 0.9, std::unique_ptr<Schedule> schedule = nullptr);

    void update(Matrix &value, const Matrix &gradient) override;

    double momentum;
    std::unordered_map<const Matrix *, Matrix> velocities;
};

}
