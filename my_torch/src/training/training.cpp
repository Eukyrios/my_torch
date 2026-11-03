/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A training needs a loss, an optimizer and a batch size of at least 1; each run continues the
** epoch count, so the learning-rate schedule picks up where the last run stopped
*/

#include <stdexcept>
#include <string>

#include "training/training.hpp"

namespace my_torch {

static void check_training(const Loss *loss, const Optimizer *optimizer, std::size_t batch_size)
{
    if (!loss || !optimizer)
        throw std::invalid_argument("a training needs a loss and an optimizer");
    if (batch_size < 1)
        throw std::invalid_argument("the batch size must be at least 1, got " + std::to_string(batch_size));
}

Training::Training(std::unique_ptr<Loss> loss, std::unique_ptr<Optimizer> optimizer, std::size_t batch_size, std::mt19937 rng)
:
    loss(std::move(loss)), optimizer(std::move(optimizer)), batch_size(batch_size), rng(rng)
{
    check_training(this->loss.get(), this->optimizer.get(), batch_size);
}

std::vector<double> Training::run(Network &network, const Matrix &inputs, const Matrix &labels, std::size_t epochs)
{
    std::vector<double> run_losses = network.train(inputs, labels, *loss, *optimizer, epochs, batch_size, rng, epoch);
    losses.insert(losses.end(), run_losses.begin(), run_losses.end());
    epoch += epochs;
    return run_losses;
}

}
