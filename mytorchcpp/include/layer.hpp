/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Layer is the base class of every layer: a forward pass on a batch, also run by calling it
*/

#pragma once

#include "matrix.hpp"

namespace my_torch {

class Layer {
public:
    virtual ~Layer() = default;

    virtual Matrix forward(const Matrix &inputs) = 0;
    Matrix operator()(const Matrix &inputs);
};

}
