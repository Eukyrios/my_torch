/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** FitScore trains a ReLU network the way --train does (early stopping, best weights kept, dropout and
** L2 when asked) and scores it by its error on held-out boards; config writes it as a generator file
*/

#pragma once

#include <cstddef>
#include <random>

#include "io/json/json.hpp"
#include "model/network/network.hpp"
#include "training/hyperparameters/search_space/search_space.hpp"
#include "training/hyperparameters/validation_loss/validation_loss.hpp"
#include "training/training.hpp"

namespace my_torch {

class FitScore {
public:
    FitScore(const TrainingData &data, std::size_t max_epochs, unsigned int seed = 0, std::size_t patience = 5);

    static Hyperparameters defaults();

    Network build(const Hyperparameters &hyperparameters, std::mt19937 &rng) const;
    Training training(const Hyperparameters &hyperparameters) const;
    Network train(const Hyperparameters &hyperparameters) const;
    double operator()(const Hyperparameters &hyperparameters) const;
    Json config(const Hyperparameters &hyperparameters, std::size_t max_epochs) const;

    TrainingData data;
    std::size_t max_epochs;
    unsigned int seed;
    std::size_t patience;
};

}
