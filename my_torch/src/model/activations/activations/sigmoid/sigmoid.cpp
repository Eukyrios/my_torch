/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Sigmoid squashes z into (0, 1), computed as 0.5 * (1 + tanh(z / 2)) so exp never overflows;
** its derivative is sigmoid * (1 - sigmoid)
*/

#include <cmath>

#include "model/activations/activations/sigmoid/sigmoid.hpp"

namespace my_torch {

static double sigmoid_value(double z)
{
    return 0.5 * (1.0 + std::tanh(0.5 * z));
}

static double sigmoid_derivative_value(double z)
{
    double sigmoid = sigmoid_value(z);
    return sigmoid * (1.0 - sigmoid);
}

Sigmoid::Sigmoid()
:
    Activation("sigmoid")
{}

Matrix Sigmoid::forward(const Matrix &z) const
{
    return z.apply(sigmoid_value);
}

Matrix Sigmoid::derivative(const Matrix &z) const
{
    return z.apply(sigmoid_derivative_value);
}

}
