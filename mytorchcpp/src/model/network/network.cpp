/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A network: its layers one after the other, each one's outputs the next one's inputs
*/

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

#include "model/network/network.hpp"

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

static void check_training_data(const Matrix &inputs, const Matrix &labels)
{
    if (inputs.rows() != labels.rows())
        throw std::invalid_argument("expected one label per sample, got " + std::to_string(inputs.rows()) + " samples and " + std::to_string(labels.rows()) + " labels");
}

static void check_training_settings(std::size_t epochs, std::size_t batch_size)
{
    if (epochs < 1 || batch_size < 1)
        throw std::invalid_argument("epochs and batch size must be at least 1, got " + std::to_string(epochs) + " and " + std::to_string(batch_size));
}

std::vector<std::vector<std::size_t>> shuffled_batches(std::size_t n_samples, std::size_t batch_size, std::mt19937 &rng)
{
    std::vector<std::size_t> order(n_samples);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<std::vector<std::size_t>> batches;
    for (std::size_t start = 0; start < n_samples; start += batch_size)
        batches.emplace_back(order.begin() + start, order.begin() + std::min(start + batch_size, n_samples));
    return batches;
}

Network::Network(std::vector<std::unique_ptr<Layer>> layers)
:
    layers(std::move(layers))
{
    check_layers(this->layers);
}

Shape Network::shape() const
{
    std::vector<Layer *> sized = find_sized_layers(layers);
    return Shape(sized.front()->shape()->first, sized.back()->shape()->second);
}

std::vector<Parameter> Network::parameters()
{
    std::vector<Parameter> all;
    for (const std::unique_ptr<Layer> &layer : layers)
        for (const Parameter &parameter : layer->parameters())
            all.push_back(parameter);
    return all;
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

Matrix Network::backward(const Matrix &gradient)
{
    Matrix result = gradient;
    for (auto layer = layers.rbegin(); layer != layers.rend(); layer++)
        result = (*layer)->backward(result);
    return result;
}

std::vector<double> Network::train(const Matrix &inputs, const Matrix &labels, const Loss &loss, Optimizer &optimizer, std::size_t epochs, std::size_t batch_size, std::mt19937 &rng)
{
    check_training_data(inputs, labels);
    check_training_settings(epochs, batch_size);
    std::vector<double> losses;
    for (std::size_t epoch = 0; epoch < epochs; epoch++)
        losses.push_back(train_epoch(inputs, labels, loss, optimizer, batch_size, rng, epoch));
    return losses;
}

double Network::train_epoch(const Matrix &inputs, const Matrix &labels, const Loss &loss, Optimizer &optimizer, std::size_t batch_size, std::mt19937 &rng, std::size_t epoch)
{
    optimizer.start_epoch(epoch);
    double total = 0.0;
    for (const std::vector<std::size_t> &batch : shuffled_batches(inputs.rows(), batch_size, rng))
        total += train_batch(inputs.select_rows(batch), labels.select_rows(batch), loss, optimizer) * batch.size();
    return total / inputs.rows();
}

double Network::train_batch(const Matrix &inputs, const Matrix &labels, const Loss &loss, Optimizer &optimizer)
{
    Matrix predictions = forward(inputs);
    backward(loss.gradient(predictions, labels));
    optimizer.step(parameters());
    return loss(predictions, labels);
}

}
