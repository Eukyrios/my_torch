/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Dropout draws a new mask at each training forward pass and backward lets the gradient through
** the inputs that were kept only; the rate must be in [0, 1)
*/

#include <stdexcept>
#include <string>

#include "model/layers/layers/dropout/dropout.hpp"

namespace my_torch {

static void check_rate(double rate)
{
    if (!(rate >= 0.0 && rate < 1.0))
        throw std::invalid_argument("the dropout rate must be in [0, 1), got " + std::to_string(rate));
}

Dropout::Dropout(double rate, std::mt19937 rng)
:
    Layer("dropout"), rate(rate), rng(rng)
{
    check_rate(rate);
}

Matrix Dropout::forward(const Matrix &inputs)
{
    mask = Matrix();
    if (!training || rate == 0.0)
        return inputs;
    std::bernoulli_distribution keep(1.0 - rate);
    mask = Matrix(inputs.rows(), inputs.cols());
    for (std::size_t row = 0; row < inputs.rows(); row++)
        for (std::size_t col = 0; col < inputs.cols(); col++)
            mask(row, col) = keep(rng) ? 1.0 / (1.0 - rate) : 0.0;
    return inputs * mask;
}

Matrix Dropout::backward(const Matrix &gradient)
{
    if (mask.size() == 0)
        return gradient;
    if (gradient.rows() != mask.rows() || gradient.cols() != mask.cols())
        throw std::invalid_argument("the gradient must have the shape of the last outputs " + std::to_string(mask.rows()) + "x" + std::to_string(mask.cols()) + ", got " + std::to_string(gradient.rows()) + "x" + std::to_string(gradient.cols()));
    return gradient * mask;
}

}
