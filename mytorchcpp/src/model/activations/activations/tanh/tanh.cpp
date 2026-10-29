/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tanh squashes z into (-1, 1): its derivative is 1 - tanh^2
*/

#include <cmath>

#include "model/activations/activations/tanh/tanh.hpp"

namespace my_torch {

static double tanh_value(double z)
{
    return std::tanh(z);
}

static double tanh_derivative_value(double z)
{
    double tanh = std::tanh(z);
    return 1.0 - tanh * tanh;
}

Tanh::Tanh()
:
    Activation("tanh")
{}

Matrix Tanh::forward(const Matrix &z) const
{
    return z.apply(tanh_value);
}

Matrix Tanh::derivative(const Matrix &z) const
{
    return z.apply(tanh_derivative_value);
}

}
