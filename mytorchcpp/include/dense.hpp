/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A dense layer: column j of the weights (n_inputs x n_outputs) belongs to neuron j, and a
** batch with one sample per row gives z = inputs * weights + biases, then its activation;
** the last inputs and z are kept for backpropagation
*/

#pragma once

#include <cstddef>
#include <vector>

#include "activations.hpp"
#include "matrix.hpp"

namespace my_torch {

class Dense {
public:
    Dense(const Matrix &weights, const std::vector<double> &biases, const Activation &activation = LINEAR);

    std::size_t n_inputs() const;
    std::size_t n_outputs() const;
    Matrix forward(const Matrix &inputs);

    Matrix weights;
    Matrix biases;
    Activation activation;
    Matrix inputs;
    Matrix z;
};

}
