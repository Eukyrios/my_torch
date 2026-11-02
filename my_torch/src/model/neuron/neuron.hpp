/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A single neuron holding one weight per input and a bias; its forward pass is
** z = x1*w1 + ... + xn*wn + b, also run by calling the neuron, neuron(inputs)
*/

#pragma once

#include <cstddef>
#include <vector>

namespace my_torch {

class Neuron {
public:
    Neuron(const std::vector<double> &weights, double bias);

    std::size_t n_inputs() const;
    double forward(const std::vector<double> &inputs) const;
    double operator()(const std::vector<double> &inputs) const;

    std::vector<double> weights;
    double bias;
};

}
