/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A dense layer: one matrix product runs every neuron on every sample of the batch
*/

#include <stdexcept>
#include <string>

#include "dense_layer.hpp"

namespace my_torch {

static void check_parameters(const Matrix &weights, const std::vector<double> &biases)
{
    if (weights.rows() == 0 || weights.cols() == 0)
        throw std::invalid_argument("weights must be a matrix (n_inputs x n_outputs), got an empty one");
    if (biases.size() != weights.cols())
        throw std::invalid_argument("expected " + std::to_string(weights.cols()) + " biases, one per neuron, got " + std::to_string(biases.size()));
}

static void check_input_size(const Matrix &inputs, std::size_t n_inputs)
{
    if (inputs.cols() != n_inputs)
        throw std::invalid_argument("the layer expects " + std::to_string(n_inputs) + " inputs per sample, got " + std::to_string(inputs.cols()));
}

DenseLayer::DenseLayer(const Matrix &weights, const std::vector<double> &biases, std::unique_ptr<Activation> activation) : Layer("dense"), weights(weights), biases(Matrix::row(biases)), activation(activation ? std::move(activation) : std::make_unique<Linear>())
{
    check_parameters(this->weights, biases);
}

std::size_t DenseLayer::n_inputs() const
{
    return weights.rows();
}

std::size_t DenseLayer::n_outputs() const
{
    return weights.cols();
}

Matrix DenseLayer::forward(const Matrix &inputs)
{
    check_input_size(inputs, n_inputs());
    this->inputs = inputs;
    z = inputs * weights + biases;
    return (*activation)(z);
}

}
