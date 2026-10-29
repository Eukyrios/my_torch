/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** ReLU cuts negative values to 0: its derivative is 1 for positive values, 0 otherwise
*/

#include <algorithm>

#include "model/activations/activations/relu/relu.hpp"

namespace my_torch {

static double relu_value(double z)
{
    return std::max(0.0, z);
}

static double relu_derivative_value(double z)
{
    return z > 0.0 ? 1.0 : 0.0;
}

ReLU::ReLU()
:
    Activation("relu")
{}

Matrix ReLU::forward(const Matrix &z) const
{
    return z.apply(relu_value);
}

Matrix ReLU::derivative(const Matrix &z) const
{
    return z.apply(relu_derivative_value);
}

}
