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

#include "activations/activation.hpp"

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

static double relu_derivative_value(double z)
{
    return z > 0.0 ? 1.0 : 0.0;
}

static double sigmoid_derivative_value(double z)
{
    double sigmoid = sigmoid_value(z);
    return sigmoid * (1.0 - sigmoid);
}

static double tanh_derivative_value(double z)
{
    double tanh = std::tanh(z);
    return 1.0 - tanh * tanh;
}

Activation::Activation(const std::string &name) : name(name)
{
}

Matrix Activation::backward(const Matrix &z, const Matrix &gradient) const
{
    return gradient.hadamard(derivative(z));
}

Matrix Activation::operator()(const Matrix &z) const
{
    return forward(z);
}

Linear::Linear() : Activation("linear")
{
}

Matrix Linear::forward(const Matrix &z) const
{
    return z;
}

Matrix Linear::derivative(const Matrix &z) const
{
    return Matrix(z.rows(), z.cols(), 1.0);
}

ReLU::ReLU() : Activation("relu")
{
}

Matrix ReLU::forward(const Matrix &z) const
{
    return z.apply(relu_value);
}

Matrix ReLU::derivative(const Matrix &z) const
{
    return z.apply(relu_derivative_value);
}

Sigmoid::Sigmoid() : Activation("sigmoid")
{
}

Matrix Sigmoid::forward(const Matrix &z) const
{
    return z.apply(sigmoid_value);
}

Matrix Sigmoid::derivative(const Matrix &z) const
{
    return z.apply(sigmoid_derivative_value);
}

Tanh::Tanh() : Activation("tanh")
{
}

Matrix Tanh::forward(const Matrix &z) const
{
    return z.apply(tanh_value);
}

Matrix Tanh::derivative(const Matrix &z) const
{
    return z.apply(tanh_derivative_value);
}

Softmax::Softmax() : Activation("softmax")
{
}

Matrix Softmax::forward(const Matrix &z) const
{
    Matrix exps = (z - z.row_maxima()).apply(std::exp);
    return exps / exps.row_sums();
}

Matrix Softmax::derivative(const Matrix &z) const
{
    Matrix softmax = forward(z);
    return softmax - softmax.hadamard(softmax);
}

Matrix Softmax::backward(const Matrix &z, const Matrix &gradient) const
{
    Matrix softmax = forward(z);
    return softmax.hadamard(gradient - gradient.hadamard(softmax).row_sums());
}

template <typename Subclass>
static std::unique_ptr<Activation> make()
{
    return std::make_unique<Subclass>();
}

const std::vector<ActivationMaker> &Activation::subclasses()
{
    static const std::vector<ActivationMaker> all = {make<Linear>, make<ReLU>, make<Sigmoid>, make<Tanh>, make<Softmax>};
    return all;
}

std::vector<std::string> Activation::names()
{
    std::vector<std::string> names;
    for (ActivationMaker make_activation : subclasses())
        names.push_back(make_activation()->name);
    return names;
}

static std::string joined(const std::vector<std::string> &names)
{
    std::string text;
    for (const std::string &name : names)
        text += text.empty() ? name : ", " + name;
    return text;
}

std::unique_ptr<Activation> Activation::from_name(const std::string &name)
{
    for (ActivationMaker make_activation : subclasses()) {
        std::unique_ptr<Activation> activation = make_activation();
        if (activation->name == name)
            return activation;
    }
    throw std::invalid_argument("unknown activation '" + name + "', expected one of: " + joined(names()));
}

}
