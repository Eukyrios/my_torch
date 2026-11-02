/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Layer is the base class of every layer: a name, used to save and reload a network, its shape
** (n_inputs, n_outputs), empty when it works on any size, forward (also run by calling it),
** backward returning the input gradient, and its (value, gradient) parameters, none by default
*/

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "math/matrix.hpp"

namespace my_torch {

using Shape = std::pair<std::size_t, std::size_t>;

struct Parameter {
    Matrix *value;
    Matrix *gradient;
};

class Layer {
public:
    explicit Layer(const std::string &name);
    virtual ~Layer() = default;

    virtual std::optional<Shape> shape() const;
    virtual std::vector<Parameter> parameters();
    virtual Matrix forward(const Matrix &inputs) = 0;
    virtual Matrix backward(const Matrix &gradient) = 0;
    Matrix operator()(const Matrix &inputs);

    std::string name;
};

}
