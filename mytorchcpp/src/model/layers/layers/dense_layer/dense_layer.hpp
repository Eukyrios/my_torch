/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A dense layer: column j of the weights (n_inputs x n_outputs) belongs to neuron j, and a
** batch gives z = inputs.matmul(weights) + biases, then its activation; create() starts from random
** weights (He for ReLU, Xavier otherwise) and backward stores the weight and bias gradients
*/

#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "model/activations/activation.hpp"
#include "model/layers/layer.hpp"
#include "math/matrix.hpp"

namespace my_torch {

class DenseLayer : public Layer {
public:
    DenseLayer(const Matrix &weights, const std::vector<double> &biases, std::unique_ptr<Activation> activation = nullptr);

    static std::unique_ptr<DenseLayer> create(std::size_t n_inputs, std::size_t n_outputs, std::unique_ptr<Activation> activation, std::mt19937 &rng);

    std::optional<Shape> shape() const override;
    std::vector<Parameter> parameters() override;
    Matrix forward(const Matrix &inputs) override;
    Matrix backward(const Matrix &gradient) override;

    Matrix weights;
    Matrix biases;
    std::unique_ptr<Activation> activation;
    Matrix inputs;
    Matrix z;
    Matrix weights_gradient;
    Matrix biases_gradient;
};

}
