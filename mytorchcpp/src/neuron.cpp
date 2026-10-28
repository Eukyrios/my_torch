/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A single neuron: the weighted sum of its inputs plus a bias
*/

#include <numeric>
#include <stdexcept>
#include <string>

#include "neuron.hpp"

namespace my_torch {

static void check_one_weight_per_input(const std::vector<double> &inputs, const std::vector<double> &weights)
{
    if (inputs.size() != weights.size())
        throw std::invalid_argument("a neuron needs one weight per input, got " + std::to_string(inputs.size()) + " inputs and " + std::to_string(weights.size()) + " weights");
}

double weighted_sum(const std::vector<double> &inputs, const std::vector<double> &weights, double bias)
{
    check_one_weight_per_input(inputs, weights);
    return std::inner_product(inputs.begin(), inputs.end(), weights.begin(), bias);
}

}
