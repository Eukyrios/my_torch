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

Matrix Activation::operator()(const Matrix &z) const
{
    return function(z);
}

const Activation LINEAR = {"linear", linear};
const Activation RELU = {"relu", relu};
const Activation SIGMOID = {"sigmoid", sigmoid};
const Activation TANH = {"tanh", tanh};
const Activation SOFTMAX = {"softmax", softmax};

const std::vector<Activation> &all_activations()
{
    static const std::vector<Activation> all = {LINEAR, RELU, SIGMOID, TANH, SOFTMAX};
    return all;
}

std::vector<std::string> activation_names()
{
    std::vector<std::string> names;
    for (const Activation &activation : all_activations())
        names.push_back(activation.name);
    return names;
}

static std::string joined(const std::vector<std::string> &names)
{
    std::string text;
    for (const std::string &name : names)
        text += text.empty() ? name : ", " + name;
    return text;
}

Activation find_activation(const std::string &name)
{
    for (const Activation &activation : all_activations())
        if (activation.name == name)
            return activation;
    throw std::invalid_argument("unknown activation '" + name + "', expected one of: " + joined(activation_names()));
}

}
