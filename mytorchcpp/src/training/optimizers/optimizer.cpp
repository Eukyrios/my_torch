/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Optimizers update every parameter in place from its gradient; the default schedule is Constant
*/

#include <stdexcept>
#include <string>

#include "training/schedules/schedules/constant/constant.hpp"
#include "training/optimizers/optimizer.hpp"

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

Optimizer::Optimizer(const std::string &name, double learning_rate, std::unique_ptr<Schedule> schedule)
:
    name(name), initial_learning_rate(learning_rate), learning_rate(learning_rate), schedule(schedule ? std::move(schedule) : std::make_unique<Constant>())
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

}
