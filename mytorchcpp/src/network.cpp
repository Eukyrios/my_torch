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

static void check_sizes_chain(const std::vector<std::unique_ptr<Layer>> &layers)
{
    std::optional<std::size_t> received;
    for (std::size_t position = 0; position < layers.size(); position++) {
        std::optional<Shape> shape = layers[position]->shape();
        if (!shape)
            continue;
        if (received && shape->first != *received)
            throw std::invalid_argument("layer " + std::to_string(position) + " expects " + std::to_string(shape->first) + " inputs but receives " + std::to_string(*received));
        received = shape->second;
    }
}

static void check_layers(const std::vector<std::unique_ptr<Layer>> &layers)
{
    for (const std::unique_ptr<Layer> &layer : layers)
        if (!layer)
            throw std::invalid_argument("a network is made of layers, got an empty one");
    if (find_sized_layers(layers).empty())
        throw std::invalid_argument("a network needs at least one layer with a fixed size, such as a dense layer");
    check_sizes_chain(layers);
}

Network::Network(std::vector<std::unique_ptr<Layer>> layers) : layers(std::move(layers))
{
    check_layers(this->layers);
}

Shape Network::shape() const
{
    std::vector<Layer *> sized = find_sized_layers(layers);
    return Shape(sized.front()->shape()->first, sized.back()->shape()->second);
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
