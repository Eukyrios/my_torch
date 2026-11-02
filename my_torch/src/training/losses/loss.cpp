/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Calling a loss runs its forward; every loss first checks that predictions and expected
** outputs have the same shape
*/

#include <stdexcept>
#include <string>

#include "training/losses/loss.hpp"

namespace my_torch {

void Loss::check_same_shape(const Matrix &predictions, const Matrix &expected)
{
    if (predictions.rows() != expected.rows() || predictions.cols() != expected.cols())
        throw std::invalid_argument("predictions and expected outputs must have the same shape, got " + std::to_string(predictions.rows()) + "x" + std::to_string(predictions.cols()) + " and " + std::to_string(expected.rows()) + "x" + std::to_string(expected.cols()));
}

Loss::Loss(const std::string &name)
:
    name(name)
{}

double Loss::operator()(const Matrix &predictions, const Matrix &expected) const
{
    return forward(predictions, expected);
}

}
