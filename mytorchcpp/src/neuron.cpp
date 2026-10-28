/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A single neuron: the weighted sum of its inputs plus its bias
*/

#include <numeric>
#include <stdexcept>
#include <string>

#include "neuron.hpp"

namespace my_torch {

static void check_parameters(const std::vector<double> &weights)
{
    if (weights.empty())
        throw std::invalid_argument("a neuron needs one weight per input, got none");
}

static void check_input_size(const std::vector<double> &inputs, std::size_t n_inputs)
{
    if (inputs.size() != n_inputs)
        throw std::invalid_argument("the neuron needs one input per weight (" + std::to_string(n_inputs) + "), got " + std::to_string(inputs.size()));
}

Neuron::Neuron(const std::vector<double> &weights, double bias) : weights(weights), bias(bias)
{
    check_parameters(this->weights);
}

std::size_t Neuron::n_inputs() const
{
    return weights.size();
}

double Neuron::forward(const std::vector<double> &inputs) const
{
    check_input_size(inputs, n_inputs());
    return std::inner_product(inputs.begin(), inputs.end(), weights.begin(), bias);
}

}
