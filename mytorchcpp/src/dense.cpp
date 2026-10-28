/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A dense layer: one weighted sum per neuron, all from the same inputs
*/

#include <stdexcept>
#include <string>

#include "dense.hpp"

namespace my_torch {

static void check_weights_are_a_matrix(const std::vector<std::vector<double>> &weights)
{
    if (weights.empty() || weights[0].empty())
        throw std::invalid_argument("weights must be a matrix (n_inputs x n_outputs), got an empty one");
    for (const std::vector<double> &row : weights)
        if (row.size() != weights[0].size())
            throw std::invalid_argument("weights must be a matrix (n_inputs x n_outputs), got rows of different sizes");
}

static void check_one_bias_per_neuron(const std::vector<std::vector<double>> &weights, const std::vector<double> &biases)
{
    if (biases.size() != weights[0].size())
        throw std::invalid_argument("expected " + std::to_string(weights[0].size()) + " biases, one per neuron, got " + std::to_string(biases.size()));
}

static void check_input_size(const std::vector<double> &inputs, std::size_t n_inputs)
{
    if (inputs.size() != n_inputs)
        throw std::invalid_argument("the layer expects " + std::to_string(n_inputs) + " inputs, got " + std::to_string(inputs.size()));
}

Dense::Dense(const std::vector<std::vector<double>> &weights, const std::vector<double> &biases) : weights(weights), biases(biases)
{
    check_weights_are_a_matrix(this->weights);
    check_one_bias_per_neuron(this->weights, this->biases);
}

std::size_t Dense::n_inputs() const
{
    return weights.size();
}

std::size_t Dense::n_outputs() const
{
    return weights[0].size();
}

std::vector<double> Dense::forward(const std::vector<double> &inputs) const
{
    check_input_size(inputs, n_inputs());
    std::vector<double> outputs = biases;
    for (std::size_t i = 0; i < n_inputs(); i++)
        for (std::size_t j = 0; j < n_outputs(); j++)
            outputs[j] += inputs[i] * weights[i][j];
    return outputs;
}

}
