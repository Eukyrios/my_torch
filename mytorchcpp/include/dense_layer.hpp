/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A dense layer: column j of the weights (n_inputs x n_outputs) belongs to neuron j, and a
** batch with one sample per row gives z = inputs * weights + biases, then its activation;
** the last inputs and z are kept for backpropagation
*/

#pragma once

#include <memory>
#include <vector>

#include "activations.hpp"
#include "layer.hpp"
#include "matrix.hpp"

namespace my_torch {

class DenseLayer : public Layer {
public:
    DenseLayer(const Matrix &weights, const std::vector<double> &biases, std::unique_ptr<Activation> activation = nullptr);

    std::optional<Shape> shape() const override;
    Matrix forward(const Matrix &inputs) override;

    Matrix weights;
    Matrix biases;
    std::unique_ptr<Activation> activation;
    Matrix inputs;
    Matrix z;
};

}
