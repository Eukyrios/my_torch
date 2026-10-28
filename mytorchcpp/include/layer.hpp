/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Layer is the base class of every layer: a name, used to save and reload a network, its
** shape (n_inputs, n_outputs), empty for layers that work on any size, and a forward pass
*/

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <utility>

#include "matrix.hpp"

namespace my_torch {

using Shape = std::pair<std::size_t, std::size_t>;

class Layer {
public:
    explicit Layer(const std::string &name);
    virtual ~Layer() = default;

    virtual std::optional<Shape> shape() const;
    virtual Matrix forward(const Matrix &inputs) = 0;
    Matrix operator()(const Matrix &inputs);

    std::string name;
};

}
