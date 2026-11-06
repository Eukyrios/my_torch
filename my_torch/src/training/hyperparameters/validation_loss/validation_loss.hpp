/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** ValidationLoss trains the network a configuration describes (hidden activation and loss included)
** and scores it on held-out data: its validation loss, or with ValidationError its validation error
*/

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "math/matrix.hpp"
#include "model/network/network.hpp"
#include "training/losses/loss.hpp"
#include "training/optimizers/optimizer.hpp"
#include "training/hyperparameters/search_space/search_space.hpp"

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
    static const std::vector<std::string> &activation_choices();
    static const std::vector<std::string> &loss_choices();
    static std::string activation_name(const Hyperparameters &hyperparameters);
    static std::string loss_name(const Hyperparameters &hyperparameters);

    Network build(const Hyperparameters &hyperparameters, std::mt19937 &rng) const;
    std::unique_ptr<Loss> loss(const Hyperparameters &hyperparameters) const;
    std::unique_ptr<Optimizer> optimizer(const Hyperparameters &hyperparameters) const;
    std::size_t batch_size(const Hyperparameters &hyperparameters) const;
    Network train(const Hyperparameters &hyperparameters) const;
    double operator()(const Hyperparameters &hyperparameters) const;
    double error(const Hyperparameters &hyperparameters) const;

    TrainingData data;
    std::size_t epochs;
    unsigned int seed;
};

class ValidationError {
public:
    explicit ValidationError(const ValidationLoss &score);

    double operator()(const Hyperparameters &hyperparameters) const;

private:
    const ValidationLoss *_score;
};

double accuracy(const Matrix &predictions, const Matrix &labels);

}
