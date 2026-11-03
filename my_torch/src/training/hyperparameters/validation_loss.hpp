/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** ValidationLoss scores one configuration: build a ReLU network ending in softmax, train it with
** Momentum and cross-entropy, return the loss on data it never trained on; usable as an Objective
*/

#pragma once

#include <cstddef>

#include "math/matrix.hpp"
#include "model/network/network.hpp"
#include "training/hyperparameters/search_space.hpp"

namespace my_torch {

struct TrainingData {
    Matrix train_inputs;
    Matrix train_labels;
    Matrix validation_inputs;
    Matrix validation_labels;
};

class ValidationLoss {
public:
    ValidationLoss(const TrainingData &data, std::size_t epochs, unsigned int seed = 0);

    static Hyperparameters defaults();

    Network build(const Hyperparameters &hyperparameters, std::mt19937 &rng) const;
    Network train(const Hyperparameters &hyperparameters) const;
    double operator()(const Hyperparameters &hyperparameters) const;

    TrainingData data;
    std::size_t epochs;
    unsigned int seed;
};

}
