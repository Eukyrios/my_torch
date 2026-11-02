/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** The base backward multiplies the gradient by the derivative; subclasses() lists every
** activation so from_name can find one by its name
*/

#include <stdexcept>

#include "model/activations/activation.hpp"
#include "model/activations/activations/linear/linear.hpp"
#include "model/activations/activations/relu/relu.hpp"
#include "model/activations/activations/sigmoid/sigmoid.hpp"
#include "model/activations/activations/softmax/softmax.hpp"
#include "model/activations/activations/tanh/tanh.hpp"

namespace my_torch {

Activation::Activation(const std::string &name)
:
    name(name)
{}

Matrix Activation::backward(const Matrix &z, const Matrix &gradient) const
{
    return gradient * derivative(z);
}

Matrix Activation::operator()(const Matrix &z) const
{
    return forward(z);
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
