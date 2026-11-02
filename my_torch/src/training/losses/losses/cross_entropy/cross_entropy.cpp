/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** CrossEntropy is -sum(expected * log(predictions)) per sample, averaged over the batch
*/

#include <cmath>
#include <stdexcept>
#include <string>

#include "training/losses/losses/cross_entropy/cross_entropy.hpp"

namespace my_torch {

static void check_epsilon(double epsilon)
{
    if (!(epsilon > 0.0 && epsilon < 1.0))
        throw std::invalid_argument("epsilon must be between 0 and 1, got " + std::to_string(epsilon));
}

CrossEntropy::CrossEntropy(double epsilon)
:
    Loss("cross_entropy"), epsilon(epsilon)
{
    check_epsilon(epsilon);
}

double CrossEntropy::forward(const Matrix &predictions, const Matrix &expected) const
{
    check_same_shape(predictions, expected);
    Matrix clipped = predictions.clip(epsilon, 1.0);
    return -(expected * clipped.apply(std::log)).sum() / predictions.rows();
}

Matrix CrossEntropy::gradient(const Matrix &predictions, const Matrix &expected) const
{
    check_same_shape(predictions, expected);
    Matrix clipped = predictions.clip(epsilon, 1.0);
    return expected / clipped * (-1.0 / predictions.rows());
}

}
