/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A network: its layers one after the other, each one's outputs the next one's inputs
*/

#include <stdexcept>
#include <string>

#include "network.hpp"

namespace my_torch {

static std::vector<DenseLayer *> find_dense_layers(const std::vector<std::unique_ptr<Layer>> &layers)
{
    std::vector<DenseLayer *> dense_layers;
    for (const std::unique_ptr<Layer> &layer : layers)
        if (DenseLayer *dense = dynamic_cast<DenseLayer *>(layer.get()))
            dense_layers.push_back(dense);
    return dense_layers;
}

static void check_sizes_chain(const std::vector<DenseLayer *> &dense_layers)
{
    for (std::size_t position = 0; position + 1 < dense_layers.size(); position++)
        if (dense_layers[position]->n_outputs() != dense_layers[position + 1]->n_inputs())
            throw std::invalid_argument("dense layer " + std::to_string(position) + " gives " + std::to_string(*dense_layers[position]->n_outputs()) + " outputs but dense layer " + std::to_string(position + 1) + " expects " + std::to_string(*dense_layers[position + 1]->n_inputs()) + " inputs");
}

static void check_layers(const std::vector<std::unique_ptr<Layer>> &layers)
{
    for (const std::unique_ptr<Layer> &layer : layers)
        if (!layer)
            throw std::invalid_argument("a network is made of layers, got an empty one");
    std::vector<DenseLayer *> dense_layers = find_dense_layers(layers);
    if (dense_layers.empty())
        throw std::invalid_argument("a network needs at least one dense layer");
    check_sizes_chain(dense_layers);
}

Network::Network(std::vector<std::unique_ptr<Layer>> layers) : layers(std::move(layers))
{
    check_layers(this->layers);
}

std::vector<DenseLayer *> Network::dense_layers() const
{
    return find_dense_layers(layers);
}

std::size_t Network::n_inputs() const
{
    return *dense_layers().front()->n_inputs();
}

std::size_t Network::n_outputs() const
{
    return *dense_layers().back()->n_outputs();
}

Matrix Network::forward(const Matrix &inputs)
{
    Matrix outputs = inputs;
    for (const std::unique_ptr<Layer> &layer : layers)
        outputs = (*layer)(outputs);
    return outputs;
}

Matrix Network::operator()(const Matrix &inputs)
{
    return forward(inputs);
}

}
