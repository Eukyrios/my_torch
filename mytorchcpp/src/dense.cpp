/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A dense layer: one matrix product runs every neuron on every sample of the batch
*/

#include <stdexcept>
#include <string>

#include "dense.hpp"

namespace my_torch {

static void check_weights_are_a_matrix(const Matrix &weights)
{
    if (weights.rows() == 0 || weights.cols() == 0)
        throw std::invalid_argument("weights must be a matrix (n_inputs x n_outputs), got an empty one");
}

static void check_one_bias_per_neuron(const Matrix &weights, const std::vector<double> &biases)
{
    if (biases.size() != weights.cols())
        throw std::invalid_argument("expected " + std::to_string(weights.cols()) + " biases, one per neuron, got " + std::to_string(biases.size()));
}

static void check_input_size(const Matrix &inputs, std::size_t n_inputs)
{
    if (inputs.cols() != n_inputs)
        throw std::invalid_argument("the layer expects " + std::to_string(n_inputs) + " inputs per sample, got " + std::to_string(inputs.cols()));
}

Dense::Dense(const Matrix &weights, const std::vector<double> &biases) : weights(weights), biases(Matrix::row(biases))
{
    check_weights_are_a_matrix(this->weights);
    check_one_bias_per_neuron(this->weights, biases);
}

std::size_t Dense::n_inputs() const
{
    return weights.rows();
}

std::size_t Dense::n_outputs() const
{
    return weights.cols();
}

Matrix Dense::forward(const Matrix &inputs) const
{
    check_input_size(inputs, n_inputs());
    return inputs * weights + biases;
}

}
