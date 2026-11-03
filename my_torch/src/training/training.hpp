/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A training: how it goes (loss, optimizer with its learning rate and schedule, batch size)
** and where it stands (epochs done, their losses, the random generator shuffling the batches).
** run trains a few more epochs and carries on from there, so it can be saved with the network
** and resumed: 10 epochs, save, load, then 10 more give the same network as 20 in one go.
** fit guards against overfitting: it keeps validation_split of the boards aside, stops once the
** loss on them has not improved for patience epochs (or after max_epochs) and keeps the best
** weights, so a fit never leaves the network worse on unseen boards than it found it
*/

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include "math/matrix.hpp"
#include "model/network/network.hpp"
#include "training/losses/loss.hpp"
#include "training/optimizers/optimizer.hpp"

namespace my_torch {

struct EpochReport {
    std::size_t epoch;
    double loss;
    double validation_loss;
    bool best;
};

struct FitResult {
    std::size_t epochs;
    double initial_validation_loss;
    double best_validation_loss;
    std::size_t best_epochs;    // epochs of this fit the kept weights had, 0 if none improved
    bool stopped_early;
};

class Training {
public:
    Training(std::unique_ptr<Loss> loss, std::unique_ptr<Optimizer> optimizer, std::size_t batch_size = 32, std::mt19937 rng = std::mt19937());

    std::vector<double> run(Network &network, const Matrix &inputs, const Matrix &labels, std::size_t epochs);
    FitResult fit(Network &network, const Matrix &inputs, const Matrix &labels, const std::function<void(const EpochReport &)> &on_epoch = nullptr);

    std::unique_ptr<Loss> loss;
    std::unique_ptr<Optimizer> optimizer;
    std::size_t batch_size;
    std::mt19937 rng;
    std::size_t epoch = 0;
    std::vector<double> losses;
    double validation_split = 0.1;
    std::size_t patience = 5;
    std::size_t max_epochs = 50;
    std::vector<double> validation_losses;
};

std::vector<std::size_t> validation_rows(std::size_t n_samples, double validation_split);

}
