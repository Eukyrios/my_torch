/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** linear keeps z, relu cuts negatives to 0, sigmoid squashes into (0, 1) as
** 0.5 * (1 + tanh(z / 2)) so exp never overflows, tanh squashes into (-1, 1), and softmax
** turns each row into probabilities, minus the row maximum first for the same reason
*/

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "activations.hpp"

namespace my_torch {

static double relu_value(double z)
{
    return std::max(0.0, z);
}

static double sigmoid_value(double z)
{
    return 0.5 * (1.0 + std::tanh(0.5 * z));
}

static double tanh_value(double z)
{
    return std::tanh(z);
}

Matrix linear(const Matrix &z)
{
    return z;
}

Matrix relu(const Matrix &z)
{
    return z.apply(relu_value);
}

Matrix sigmoid(const Matrix &z)
{
    return z.apply(sigmoid_value);
}

Matrix tanh(const Matrix &z)
{
    return z.apply(tanh_value);
}

Matrix softmax(const Matrix &z)
{
    Matrix exps = (z - z.row_maxima()).apply(std::exp);
    return exps / exps.row_sums();
}

const std::map<std::string, ActivationFunction> &activations()
{
    static const std::map<std::string, ActivationFunction> all = {
        {"linear", linear},
        {"relu", relu},
        {"sigmoid", sigmoid},
        {"tanh", tanh},
        {"softmax", softmax},
    };
    return all;
}

static std::string activation_names()
{
    std::string names;
    for (const auto &[name, function] : activations())
        names += names.empty() ? name : ", " + name;
    return names;
}

ActivationFunction get_activation(const std::string &name)
{
    auto found = activations().find(name);
    if (found == activations().end())
        throw std::invalid_argument("unknown activation '" + name + "', expected one of: " + activation_names());
    return found->second;
}

}
