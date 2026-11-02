/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of a training run in several parts
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "model/activations/activations/softmax/softmax.hpp"
#include "model/activations/activations/tanh/tanh.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/schedules/schedules/step_decay/step_decay.hpp"
#include "training/training.hpp"

static const my_torch::Matrix INPUTS({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
static const my_torch::Matrix LABELS({{1, 0}, {0, 1}, {0, 1}, {1, 0}});

static my_torch::Network small_network()
{
    std::mt19937 rng(3);
    std::vector<std::unique_ptr<my_torch::Layer>> layers;
    layers.push_back(my_torch::DenseLayer::create(2, 4, std::make_unique<my_torch::Tanh>(), rng));
    layers.push_back(my_torch::DenseLayer::create(4, 2, std::make_unique<my_torch::Softmax>(), rng));
    return my_torch::Network(std::move(layers));
}

static my_torch::Training momentum_training()
{
    return my_torch::Training(std::make_unique<my_torch::CrossEntropy>(), std::make_unique<my_torch::Momentum>(0.1, 0.9, std::make_unique<my_torch::StepDecay>(0.5, 3)), 2, std::mt19937(7));
}

static const my_torch::Matrix &weights(const my_torch::Network &network, std::size_t layer)
{
    return dynamic_cast<const my_torch::DenseLayer &>(*network.layers[layer]).weights;
}

Test(training, a_run_counts_its_epochs_and_keeps_their_losses)
{
    my_torch::Network network = small_network();
    my_torch::Training training = momentum_training();

    std::vector<double> first = training.run(network, INPUTS, LABELS, 4);
    std::vector<double> second = training.run(network, INPUTS, LABELS, 3);
    cr_assert_eq(first.size(), 4);
    cr_assert_eq(second.size(), 3);
    cr_assert_eq(training.epoch, 7);
    cr_assert_eq(training.losses.size(), 7);
    cr_assert_eq(training.losses[4], second[0]);
}

Test(training, the_schedule_carries_on_between_runs)
{
    my_torch::Network network = small_network();
    my_torch::Training training = momentum_training();

    training.run(network, INPUTS, LABELS, 4);
    cr_assert_float_eq(training.optimizer->learning_rate, 0.05, 1e-12);
    training.run(network, INPUTS, LABELS, 3);
    cr_assert_float_eq(training.optimizer->learning_rate, 0.025, 1e-12);
}

Test(training, two_runs_give_the_same_network_as_one)
{
    my_torch::Network in_parts = small_network();
    my_torch::Network in_one_go = small_network();
    my_torch::Training parts = momentum_training();
    my_torch::Training one_go = momentum_training();

    parts.run(in_parts, INPUTS, LABELS, 5);
    parts.run(in_parts, INPUTS, LABELS, 5);
    one_go.run(in_one_go, INPUTS, LABELS, 10);
    for (std::size_t layer = 0; layer < 2; layer++)
        for (std::size_t row = 0; row < weights(in_one_go, layer).rows(); row++)
            for (std::size_t col = 0; col < weights(in_one_go, layer).cols(); col++)
                cr_assert_eq(weights(in_parts, layer)(row, col), weights(in_one_go, layer)(row, col));
    cr_assert(parts.losses == one_go.losses);
}

Test(training, needs_a_loss_an_optimizer_and_a_batch_size)
{
    cr_assert_throw(my_torch::Training(nullptr, std::make_unique<my_torch::Momentum>()), std::invalid_argument);
    cr_assert_throw(my_torch::Training(std::make_unique<my_torch::CrossEntropy>(), nullptr), std::invalid_argument);
    cr_assert_throw(my_torch::Training(std::make_unique<my_torch::CrossEntropy>(), std::make_unique<my_torch::Momentum>(), 0), std::invalid_argument);
}
