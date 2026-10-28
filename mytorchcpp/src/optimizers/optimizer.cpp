/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Optimizers update every parameter in place from its gradient
*/

#include <stdexcept>
#include <string>

#include "optimizers/optimizer.hpp"

namespace my_torch {

static void check_learning_rate(double learning_rate)
{
    if (!(learning_rate > 0.0))
        throw std::invalid_argument("the learning rate must be positive, got " + std::to_string(learning_rate));
}

static void check_gradient(const Matrix &gradient)
{
    if (gradient.size() == 0)
        throw std::runtime_error("a parameter has no gradient yet: run backward before step");
}

static void check_momentum(double momentum)
{
    if (!(momentum >= 0.0 && momentum < 1.0))
        throw std::invalid_argument("the momentum must be in [0, 1), got " + std::to_string(momentum));
}

Optimizer::Optimizer(const std::string &name, double learning_rate, std::unique_ptr<Schedule> schedule) : name(name), initial_learning_rate(learning_rate), learning_rate(learning_rate), schedule(schedule ? std::move(schedule) : std::make_unique<Constant>())
{
    check_learning_rate(learning_rate);
}

void Optimizer::start_epoch(std::size_t epoch)
{
    learning_rate = (*schedule)(initial_learning_rate, epoch);
}

void Optimizer::step(const std::vector<Parameter> &parameters)
{
    for (const Parameter &parameter : parameters) {
        check_gradient(*parameter.gradient);
        update(*parameter.value, *parameter.gradient);
    }
}

SGD::SGD(double learning_rate, std::unique_ptr<Schedule> schedule) : Optimizer("sgd", learning_rate, std::move(schedule))
{
}

void SGD::update(Matrix &value, const Matrix &gradient)
{
    value -= gradient * learning_rate;
}

Momentum::Momentum(double learning_rate, double momentum, std::unique_ptr<Schedule> schedule) : Optimizer("momentum", learning_rate, std::move(schedule)), momentum(momentum)
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
