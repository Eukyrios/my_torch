/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Layer is the base class of every layer: a name, used to save and reload a network, its shape
** (n_inputs, n_outputs), empty for layers that work on any size, a forward pass (also run by
** calling the layer) and a backward pass returning the input gradient
*/

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "math/matrix.hpp"

namespace my_torch {

using Shape = std::pair<std::size_t, std::size_t>;

class Layer {
public:
    explicit Layer(const std::string &name);
    virtual ~Layer() = default;

    virtual std::optional<Shape> shape() const;
    virtual Matrix forward(const Matrix &inputs) = 0;
    virtual Matrix backward(const Matrix &gradient) = 0;
    Matrix operator()(const Matrix &inputs);

    std::string name;
};

}
