/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Optimizer is the base class: a name, a learning rate and step(parameters) after each backward
** pass; SGD moves each parameter against its gradient, value -= learning_rate * gradient
*/

#pragma once

#include <string>
#include <vector>

#include "layers/layer.hpp"
#include "math/matrix.hpp"

namespace my_torch {

class Optimizer {
public:
    Optimizer(const std::string &name, double learning_rate);
    virtual ~Optimizer() = default;

    void step(const std::vector<Parameter> &parameters);
    virtual void update(Matrix &value, const Matrix &gradient) = 0;

    std::string name;
    double learning_rate;
};

class SGD : public Optimizer {
public:
    explicit SGD(double learning_rate = 0.01);

    void update(Matrix &value, const Matrix &gradient) override;
};

}
