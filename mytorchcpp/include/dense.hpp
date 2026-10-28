/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A dense layer: n_outputs neurons reading the same n_inputs values; column j of the
** weights (n_inputs x n_outputs) belongs to neuron j, outputs = inputs . weights + biases
*/

#pragma once

#include <cstddef>
#include <vector>

namespace my_torch {

class Dense {
public:
    Dense(const std::vector<std::vector<double>> &weights, const std::vector<double> &biases);

    std::size_t n_inputs() const;
    std::size_t n_outputs() const;
    std::vector<double> forward(const std::vector<double> &inputs) const;

    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
};

}
