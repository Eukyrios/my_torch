/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A single neuron: the weighted sum of its inputs plus a bias
*/

#include <numeric>

#include "neuron.hpp"

namespace my_torch {

double weighted_sum(const std::vector<double> &inputs, const std::vector<double> &weights, double bias)
{
    return std::inner_product(inputs.begin(), inputs.end(), weights.begin(), bias);
}

}
