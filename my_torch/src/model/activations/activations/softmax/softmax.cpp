/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Softmax: exp(z - max) / sum per row, and backward as s * (gradient - sum(gradient * s))
*/

#include <cmath>

#include "model/activations/activations/softmax/softmax.hpp"

namespace my_torch {

Softmax::Softmax()
:
    Activation("softmax")
{}

Matrix Softmax::forward(const Matrix &z) const
{
    Matrix exps = (z - z.row_maxima()).apply(std::exp);
    return exps / exps.row_sums();
}

Matrix Softmax::derivative(const Matrix &z) const
{
    Matrix softmax = forward(z);
    return softmax - softmax * softmax;
}

Matrix Softmax::backward(const Matrix &z, const Matrix &gradient) const
{
    Matrix softmax = forward(z);
    return softmax * (gradient - (gradient * softmax).row_sums());
}

}
