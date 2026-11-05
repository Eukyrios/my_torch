/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** He: sqrt(2 / n_inputs), suited to ReLU; Xavier: sqrt(2 / (n_inputs + n_outputs)), suited to
** tanh, sigmoid and softmax; LeCun: sqrt(1 / n_inputs), suited to tanh and sigmoid
*/

#include <cmath>
#include <stdexcept>
#include <utility>

#include "model/activations/activations/relu/relu.hpp"
#include "model/initialization/initialization.hpp"

namespace my_torch {

static const std::vector<std::pair<std::string, Initialization>> NAMES = {
    {"auto", Initialization::Auto},
    {"he", Initialization::He},
    {"xavier", Initialization::Xavier},
    {"lecun", Initialization::LeCun},
};

static std::string joined_names()
{
    std::string text;
    for (const auto &[name, initialization] : NAMES)
        text += text.empty() ? name : ", " + name;
    return text;
}

static Initialization resolved(Initialization initialization, const Activation &activation)
{
    if (initialization != Initialization::Auto)
        return initialization;
    return dynamic_cast<const ReLU *>(&activation) ? Initialization::He : Initialization::Xavier;
}

Initialization initialization_from_name(const std::string &name)
{
    for (const auto &[known, initialization] : NAMES)
        if (known == name)
            return initialization;
    throw std::invalid_argument("unknown initialization '" + name + "', expected one of: " + joined_names());
}

std::string initialization_name(Initialization initialization)
{
    for (const auto &[name, known] : NAMES)
        if (known == initialization)
            return name;
    return "auto";
}

std::vector<std::string> initialization_names()
{
    std::vector<std::string> names;
    for (const auto &[name, initialization] : NAMES)
        names.push_back(name);
    return names;
}

double initial_scale(Initialization initialization, std::size_t n_inputs, std::size_t n_outputs, const Activation &activation)
{
    switch (resolved(initialization, activation)) {
    case Initialization::He:
        return std::sqrt(2.0 / n_inputs);
    case Initialization::LeCun:
        return std::sqrt(1.0 / n_inputs);
    default:
        return std::sqrt(2.0 / (n_inputs + n_outputs));
    }
}

}
