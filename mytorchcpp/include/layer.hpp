/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Layer is the base class of every layer: a name, used to save and reload a network, and a
** forward pass on a batch, also run by calling the layer
*/

#pragma once

#include <string>

#include "matrix.hpp"

namespace my_torch {

class Layer {
public:
    explicit Layer(const std::string &name);
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix &inputs) = 0;
    Matrix operator()(const Matrix &inputs);

    std::string name;
};

}
