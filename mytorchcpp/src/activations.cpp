/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Linear keeps z, ReLU cuts negatives to 0, Sigmoid squashes into (0, 1) as
** 0.5 * (1 + tanh(z / 2)) so exp never overflows, Tanh squashes into (-1, 1), and Softmax
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

Matrix Activation::operator()(const Matrix &z) const
{
    return forward(z);
}

std::string Linear::name() const
{
    return "linear";
}

Matrix Linear::forward(const Matrix &z) const
{
    return z;
}

std::string ReLU::name() const
{
    return "relu";
}

Matrix ReLU::forward(const Matrix &z) const
{
    return z.apply(relu_value);
}

std::string Sigmoid::name() const
{
    return "sigmoid";
}

Matrix Sigmoid::forward(const Matrix &z) const
{
    return z.apply(sigmoid_value);
}

std::string Tanh::name() const
{
    return "tanh";
}

Matrix Tanh::forward(const Matrix &z) const
{
    return z.apply(tanh_value);
}

std::string Softmax::name() const
{
    return "softmax";
}

Matrix Softmax::forward(const Matrix &z) const
{
    Matrix exps = (z - z.row_maxima()).apply(std::exp);
    return exps / exps.row_sums();
}

template <typename Subclass>
static std::unique_ptr<Activation> make()
{
    return std::make_unique<Subclass>();
}

const std::vector<ActivationMaker> &activation_classes()
{
    static const std::vector<ActivationMaker> all = {make<Linear>, make<ReLU>, make<Sigmoid>, make<Tanh>, make<Softmax>};
    return all;
}

std::vector<std::string> activation_names()
{
    std::vector<std::string> names;
    for (ActivationMaker make_activation : activation_classes())
        names.push_back(make_activation()->name());
    return names;
}

static std::string joined(const std::vector<std::string> &names)
{
    std::string text;
    for (const std::string &name : names)
        text += text.empty() ? name : ", " + name;
    return text;
}

std::unique_ptr<Activation> find_activation(const std::string &name)
{
    for (ActivationMaker make_activation : activation_classes()) {
        std::unique_ptr<Activation> activation = make_activation();
        if (activation->name() == name)
            return activation;
    }
    throw std::invalid_argument("unknown activation '" + name + "', expected one of: " + joined(activation_names()));
}

}
