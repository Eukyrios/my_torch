/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A dense layer: column j of the weights (n_inputs x n_outputs) belongs to neuron j, and a
** batch with one sample per row gives outputs = inputs * weights + biases, one row per sample
*/

#pragma once

#include <cstddef>
#include <vector>

#include "matrix.hpp"

namespace my_torch {

class Dense {
public:
    Dense(const Matrix &weights, const std::vector<double> &biases);

    std::size_t n_inputs() const;
    std::size_t n_outputs() const;
    Matrix forward(const Matrix &inputs) const;

    Matrix weights;
    Matrix biases;
};

}
