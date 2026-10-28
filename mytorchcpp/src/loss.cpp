/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** MeanSquaredError is the mean of (prediction - expected)^2 over every value; CrossEntropy
** compares softmax probabilities to one-hot outputs, clipped away from 0 so log never sees 0
*/

#include <cmath>
#include <stdexcept>
#include <string>

#include "loss.hpp"

namespace my_torch {

static const double EPSILON = 1e-12;

static double square(double value)
{
    return value * value;
}

static void check_same_shape(const Matrix &predictions, const Matrix &expected)
{
    if (predictions.rows() != expected.rows() || predictions.cols() != expected.cols())
        throw std::invalid_argument("predictions and expected outputs must have the same shape, got " + std::to_string(predictions.rows()) + "x" + std::to_string(predictions.cols()) + " and " + std::to_string(expected.rows()) + "x" + std::to_string(expected.cols()));
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

CrossEntropy::CrossEntropy() : Loss("cross_entropy")
{
}

double CrossEntropy::forward(const Matrix &predictions, const Matrix &expected) const
{
    check_same_shape(predictions, expected);
    Matrix clipped = predictions.clip(EPSILON, 1.0);
    return -expected.hadamard(clipped.apply(std::log)).sum() / predictions.rows();
}

Matrix CrossEntropy::gradient(const Matrix &predictions, const Matrix &expected) const
{
    check_same_shape(predictions, expected);
    Matrix clipped = predictions.clip(EPSILON, 1.0);
    return expected / clipped * (-1.0 / predictions.rows());
}

}
