/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A network runs its layers in order on a batch, network(inputs), and backward runs them in
** reverse; train shuffles the data each epoch into mini-batches and returns each epoch's loss
*/

#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "layers/layer.hpp"
#include "losses/loss.hpp"
#include "optimizers/optimizer.hpp"

namespace my_torch {

std::vector<std::vector<std::size_t>> shuffled_batches(std::size_t n_samples, std::size_t batch_size, std::mt19937 &rng);

class Network {
public:
    explicit Network(std::vector<std::unique_ptr<Layer>> layers);

    Shape shape() const;
    std::vector<Parameter> parameters();
    Matrix forward(const Matrix &inputs);
    Matrix operator()(const Matrix &inputs);
    Matrix backward(const Matrix &gradient);
    std::vector<double> train(const Matrix &inputs, const Matrix &labels, const Loss &loss, Optimizer &optimizer, std::size_t epochs, std::size_t batch_size, std::mt19937 &rng);
    double train_epoch(const Matrix &inputs, const Matrix &labels, const Loss &loss, Optimizer &optimizer, std::size_t batch_size, std::mt19937 &rng, std::size_t epoch = 0);
    double train_batch(const Matrix &inputs, const Matrix &labels, const Loss &loss, Optimizer &optimizer);

    std::vector<std::unique_ptr<Layer>> layers;
};

}
