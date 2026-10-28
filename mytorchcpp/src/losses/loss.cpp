/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** MeanSquaredError is the mean of (prediction - expected)^2 over every value; CrossEntropy
** compares softmax probabilities to one-hot outputs, clipped to at least epsilon (1e-12 by
** default) so log never sees 0
*/

#include <cmath>
#include <stdexcept>
#include <string>

#include "losses/loss.hpp"

namespace my_torch {

static double square(double value)
{
    return value * value;
}

static void check_same_shape(const Matrix &predictions, const Matrix &expected)
{
    if (predictions.rows() != expected.rows() || predictions.cols() != expected.cols())
        throw std::invalid_argument("predictions and expected outputs must have the same shape, got " + std::to_string(predictions.rows()) + "x" + std::to_string(predictions.cols()) + " and " + std::to_string(expected.rows()) + "x" + std::to_string(expected.cols()));
}

static void check_epsilon(double epsilon)
{
    if (!(epsilon > 0.0 && epsilon < 1.0))
        throw std::invalid_argument("epsilon must be between 0 and 1, got " + std::to_string(epsilon));
}

Loss::Loss(const std::string &name) : name(name)
{
}

double Loss::operator()(const Matrix &predictions, const Matrix &expected) const
{
    return forward(predictions, expected);
}

MeanSquaredError::MeanSquaredError() : Loss("mean_squared_error")
{
}

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

CrossEntropy::CrossEntropy(double epsilon) : Loss("cross_entropy"), epsilon(epsilon)
{
    check_epsilon(epsilon);
}

double CrossEntropy::forward(const Matrix &predictions, const Matrix &expected) const
{
    check_same_shape(predictions, expected);
    Matrix clipped = predictions.clip(epsilon, 1.0);
    return -expected.hadamard(clipped.apply(std::log)).sum() / predictions.rows();
}

Matrix CrossEntropy::gradient(const Matrix &predictions, const Matrix &expected) const
{
    check_same_shape(predictions, expected);
    Matrix clipped = predictions.clip(epsilon, 1.0);
    return expected / clipped * (-1.0 / predictions.rows());
}

}
