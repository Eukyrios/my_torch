/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** MeanSquaredError and its gradient, 2 * (prediction - expected) / number of values
*/

#include "training/losses/losses/mean_squared_error/mean_squared_error.hpp"

namespace my_torch {

static double square(double value)
{
    return value * value;
}

MeanSquaredError::MeanSquaredError()
:
    Loss("mean_squared_error")
{}

double MeanSquaredError::forward(const Matrix &predictions, const Matrix &expected) const
{
    check_same_shape(predictions, expected);
    return (predictions - expected).apply(square).sum() / predictions.size();
}

Matrix MeanSquaredError::gradient(const Matrix &predictions, const Matrix &expected) const
{
    check_same_shape(predictions, expected);
    return (predictions - expected) * (2.0 / predictions.size());
}

}
