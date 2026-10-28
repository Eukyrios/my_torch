/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Layer is the base class of every layer: a name, used to save and reload a network, its
** sizes, empty for layers that work on any size, and a forward pass, also run by calling it
*/

#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include "matrix.hpp"

namespace my_torch {

class Layer {
public:
    explicit Layer(const std::string &name);
    virtual ~Layer() = default;

    virtual std::optional<std::size_t> n_inputs() const;
    virtual std::optional<std::size_t> n_outputs() const;
    virtual Matrix forward(const Matrix &inputs) = 0;
    Matrix operator()(const Matrix &inputs);

    std::string name;
};

}
