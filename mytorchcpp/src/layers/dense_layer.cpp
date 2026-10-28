/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A dense layer: one matrix product runs every neuron on every sample of the batch
*/

#include <cmath>
#include <stdexcept>
#include <string>

#include "layers/dense_layer.hpp"

namespace my_torch {

static void check_sizes(std::size_t n_inputs, std::size_t n_outputs)
{
    if (n_inputs < 1 || n_outputs < 1)
        throw std::invalid_argument("a dense layer needs at least 1 input and 1 output, got " + std::to_string(n_inputs) + " and " + std::to_string(n_outputs));
}

static double initial_scale(std::size_t n_inputs, std::size_t n_outputs, const Activation &activation)
{
    if (dynamic_cast<const ReLU *>(&activation))
        return std::sqrt(2.0 / n_inputs);
    return std::sqrt(2.0 / (n_inputs + n_outputs));
}

static Matrix random_weights(std::size_t n_inputs, std::size_t n_outputs, double scale, std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, scale);
    Matrix weights(n_inputs, n_outputs);
    for (std::size_t row = 0; row < n_inputs; row++)
        for (std::size_t col = 0; col < n_outputs; col++)
            weights(row, col) = normal(rng);
    return weights;
}

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

static void check_backward(const Matrix &z, const Matrix &gradient)
{
    if (z.size() == 0)
        throw std::runtime_error("backward needs the values of a forward pass: call forward first");
    if (gradient.rows() != z.rows() || gradient.cols() != z.cols())
        throw std::invalid_argument("the gradient must have the shape of the last outputs " + std::to_string(z.rows()) + "x" + std::to_string(z.cols()) + ", got " + std::to_string(gradient.rows()) + "x" + std::to_string(gradient.cols()));
}

DenseLayer::DenseLayer(const Matrix &weights, const std::vector<double> &biases, std::unique_ptr<Activation> activation) : Layer("dense"), weights(weights), biases(Matrix::row(biases)), activation(activation ? std::move(activation) : std::make_unique<Linear>())
{
    check_parameters(this->weights, biases);
}

std::unique_ptr<DenseLayer> DenseLayer::create(std::size_t n_inputs, std::size_t n_outputs, std::unique_ptr<Activation> activation, std::mt19937 &rng)
{
    check_sizes(n_inputs, n_outputs);
    if (!activation)
        activation = std::make_unique<Linear>();
    double scale = initial_scale(n_inputs, n_outputs, *activation);
    return std::make_unique<DenseLayer>(random_weights(n_inputs, n_outputs, scale, rng), std::vector<double>(n_outputs, 0.0), std::move(activation));
}

std::optional<Shape> DenseLayer::shape() const
{
    return Shape(weights.rows(), weights.cols());
}

std::vector<Parameter> DenseLayer::parameters()
{
    return {{&weights, &weights_gradient}, {&biases, &biases_gradient}};
}

Matrix DenseLayer::forward(const Matrix &inputs)
{
    check_input_size(inputs, weights.rows());
    this->inputs = inputs;
    z = inputs * weights + biases;
    return (*activation)(z);
}

Matrix DenseLayer::backward(const Matrix &gradient)
{
    check_backward(z, gradient);
    Matrix z_gradient = activation->backward(z, gradient);
    weights_gradient = inputs.transpose() * z_gradient;
    biases_gradient = z_gradient.column_sums();
    return z_gradient * weights.transpose();
}

}
