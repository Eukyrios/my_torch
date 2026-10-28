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

static bool is_sized(const Layer &layer)
{
    return layer.shape().has_value();
}

static std::vector<Layer *> find_sized_layers(const std::vector<std::unique_ptr<Layer>> &layers)
{
    std::vector<Layer *> sized;
    for (const std::unique_ptr<Layer> &layer : layers)
        if (is_sized(*layer))
            sized.push_back(layer.get());
    return sized;
}

static void check_sizes_chain(const std::vector<Layer *> &sized)
{
    for (std::size_t position = 0; position + 1 < sized.size(); position++)
        if (sized[position]->shape()->second != sized[position + 1]->shape()->first)
            throw std::invalid_argument("sized layer " + std::to_string(position) + " gives " + std::to_string(sized[position]->shape()->second) + " outputs but sized layer " + std::to_string(position + 1) + " expects " + std::to_string(sized[position + 1]->shape()->first) + " inputs");
}

static void check_layers(const std::vector<std::unique_ptr<Layer>> &layers)
{
    for (const std::unique_ptr<Layer> &layer : layers)
        if (!layer)
            throw std::invalid_argument("a network is made of layers, got an empty one");
    std::vector<Layer *> sized = find_sized_layers(layers);
    if (sized.empty())
        throw std::invalid_argument("a network needs at least one layer with a fixed size, such as a dense layer");
    check_sizes_chain(sized);
}

Network::Network(std::vector<std::unique_ptr<Layer>> layers) : layers(std::move(layers))
{
    check_layers(this->layers);
}

std::vector<Layer *> Network::sized_layers() const
{
    return find_sized_layers(layers);
}

std::size_t Network::n_inputs() const
{
    return sized_layers().front()->shape()->first;
}

std::size_t Network::n_outputs() const
{
    return sized_layers().back()->shape()->second;
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
