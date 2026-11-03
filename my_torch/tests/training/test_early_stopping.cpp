/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the guards against overfitting: validation split, early stopping and L2 regularisation
*/

#include <algorithm>
#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "assert_matrix.hpp"
#include "model/activations/activations/relu/relu.hpp"
#include "model/activations/activations/softmax/softmax.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/optimizers/optimizers/sgd/sgd.hpp"
#include "training/training.hpp"

struct Data {
    my_torch::Matrix inputs;
    my_torch::Matrix labels;
};

// random labels: nothing to learn, so a big enough network can only memorise, which is overfitting
static Data noise(std::size_t samples, unsigned int seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Data data {my_torch::Matrix(samples, 8), my_torch::Matrix(samples, 2)};
    for (std::size_t row = 0; row < samples; row++) {
        for (std::size_t col = 0; col < 8; col++)
            data.inputs(row, col) = normal(rng);
        data.labels(row, rng() % 2) = 1.0;
    }
    return data;
}

// the label is the sign of the first input: easy to learn
static Data signal(std::size_t samples, unsigned int seed)
{
    Data data = noise(samples, seed);
    for (std::size_t row = 0; row < samples; row++) {
        bool positive = data.inputs(row, 0) > 0;
        data.labels(row, 0) = positive;
        data.labels(row, 1) = !positive;
    }
    return data;
}

static my_torch::Network wide_network(std::size_t hidden = 64)
{
    std::mt19937 rng(2);
    std::vector<std::unique_ptr<my_torch::Layer>> layers;
    layers.push_back(my_torch::DenseLayer::create(8, hidden, std::make_unique<my_torch::ReLU>(), rng));
    layers.push_back(my_torch::DenseLayer::create(hidden, 2, std::make_unique<my_torch::Softmax>(), rng));
    return my_torch::Network(std::move(layers));
}

static my_torch::Training momentum_training()
{
    return my_torch::Training(std::make_unique<my_torch::CrossEntropy>(), std::make_unique<my_torch::Momentum>(0.05, 0.9), 8, std::mt19937(4));
}

static double loss_on(my_torch::Network &network, const Data &data, const std::vector<std::size_t> &rows)
{
    return my_torch::CrossEntropy()(network(data.inputs.select_rows(rows)), data.labels.select_rows(rows));
}

/* Validation split */

Test(validation_rows, keeps_the_share_asked_sorted_and_always_the_same)
{
    std::vector<std::size_t> rows = my_torch::validation_rows(1000, 0.1);

    cr_assert_eq(rows.size(), 100);
    cr_assert(std::is_sorted(rows.begin(), rows.end()));
    cr_assert(std::adjacent_find(rows.begin(), rows.end()) == rows.end());
    cr_assert(rows.back() < 1000);
    cr_assert(rows == my_torch::validation_rows(1000, 0.1));
    cr_assert(rows != std::vector<std::size_t>(rows.size()), "the rows are drawn, not the first ones");
}

Test(validation_rows, keeps_at_least_one_board_aside_and_one_to_train_on)
{
    cr_assert_eq(my_torch::validation_rows(5, 0.01).size(), 1);
    cr_assert_eq(my_torch::validation_rows(2, 0.9).size(), 1);
    cr_assert(my_torch::validation_rows(50, 0.0).empty());
}

/* Early stopping */

Test(early_stopping, stops_once_the_network_memorises_and_keeps_its_best_weights)
{
    Data data = noise(60, 1);
    my_torch::Network network = wide_network();
    my_torch::Training training = momentum_training();
    std::vector<std::size_t> validation = my_torch::validation_rows(60, 0.25);

    training.validation_split = 0.25;
    training.patience = 3;
    training.max_epochs = 300;
    my_torch::FitResult result = training.fit(network, data.inputs, data.labels);
    cr_assert(result.stopped_early);
    cr_assert_lt(result.epochs, 300);
    cr_assert_eq(result.epochs, result.best_epochs + 3);
    cr_assert_eq(training.validation_losses.size(), result.epochs);
    cr_assert_float_eq(loss_on(network, data, validation), result.best_validation_loss, 1e-12);
    cr_assert_leq(result.best_validation_loss, result.initial_validation_loss);
    cr_assert_eq(*std::min_element(training.validation_losses.begin(), training.validation_losses.end()), result.best_validation_loss);
}

Test(early_stopping, never_leaves_the_network_worse_on_validation)
{
    Data data = noise(60, 7);
    my_torch::Network network = wide_network();
    my_torch::Training training = momentum_training();

    training.validation_split = 0.25;
    training.patience = 2;
    training.run(network, data.inputs, data.labels, 40);
    std::vector<std::size_t> validation = my_torch::validation_rows(60, 0.25);
    double before = loss_on(network, data, validation);
    my_torch::FitResult result = training.fit(network, data.inputs, data.labels);
    cr_assert_float_eq(result.initial_validation_loss, before, 1e-12);
    cr_assert_leq(loss_on(network, data, validation), before);
}

Test(early_stopping, learns_what_can_be_learned)
{
    Data data = signal(200, 3);
    my_torch::Network network = wide_network(16);
    my_torch::Training training = momentum_training();

    training.max_epochs = 30;
    my_torch::FitResult result = training.fit(network, data.inputs, data.labels);
    cr_assert_lt(result.best_validation_loss, 0.5 * result.initial_validation_loss);
    cr_assert_gt(result.best_epochs, 0);
}

Test(early_stopping, reports_every_epoch)
{
    Data data = signal(40, 5);
    my_torch::Network network = wide_network(4);
    my_torch::Training training = momentum_training();
    std::vector<my_torch::EpochReport> reports;

    training.run(network, data.inputs, data.labels, 2);
    training.max_epochs = 4;
    training.patience = 10;
    training.fit(network, data.inputs, data.labels, [&](const my_torch::EpochReport &report) { reports.push_back(report); });
    cr_assert_eq(reports.size(), 4);
    cr_assert_eq(reports[0].epoch, 2);
    cr_assert_eq(reports[3].epoch, 5);
    cr_assert_eq(reports[3].loss, training.losses.back());
    cr_assert_eq(reports[3].validation_loss, training.validation_losses.back());
}

Test(early_stopping, without_validation_trains_every_epoch)
{
    Data data = signal(40, 5);
    my_torch::Network network = wide_network(4);
    my_torch::Training training = momentum_training();

    training.validation_split = 0.0;
    training.max_epochs = 6;
    training.patience = 1;
    my_torch::FitResult result = training.fit(network, data.inputs, data.labels);
    cr_assert_eq(result.epochs, 6);
    cr_assert_eq(result.best_epochs, 6);
    cr_assert_not(result.stopped_early);
    cr_assert(std::isnan(training.validation_losses.back()));
}

Test(early_stopping, rejects_bad_settings)
{
    Data data = signal(10, 5);
    my_torch::Network network = wide_network(4);
    my_torch::Training training = momentum_training();

    training.validation_split = 1.0;
    cr_assert_throw(training.fit(network, data.inputs, data.labels), std::invalid_argument);
    training.validation_split = 0.1;
    training.patience = 0;
    cr_assert_throw(training.fit(network, data.inputs, data.labels), std::invalid_argument);
    training.patience = 2;
    cr_assert_throw(training.fit(network, data.inputs.select_rows({0}), data.labels.select_rows({0})), std::invalid_argument);
}

/* L2 regularisation */

Test(weight_decay, adds_the_decay_to_the_gradient_of_weights_only)
{
    my_torch::Matrix weights = my_torch::Matrix::row({2, -4});
    my_torch::Matrix weights_gradient = my_torch::Matrix::row({0, 0});
    my_torch::Matrix biases = my_torch::Matrix::row({2});
    my_torch::Matrix biases_gradient = my_torch::Matrix::row({0});
    my_torch::SGD optimizer(0.5);

    optimizer.weight_decay = 0.1;
    optimizer.step({{&weights, &weights_gradient}, {&biases, &biases_gradient, false}});
    assert_matrix_eq(weights, my_torch::Matrix::row({1.9, -3.8}));
    assert_matrix_eq(biases, my_torch::Matrix::row({2}));
}

Test(weight_decay, keeps_the_weights_smaller)
{
    Data data = noise(60, 1);
    my_torch::Network free = wide_network();
    my_torch::Network decayed = wide_network();
    my_torch::Training free_training = momentum_training();
    my_torch::Training decayed_training = momentum_training();

    decayed_training.optimizer->weight_decay = 0.05;
    free_training.run(free, data.inputs, data.labels, 30);
    decayed_training.run(decayed, data.inputs, data.labels, 30);
    auto size = [](my_torch::Network &network) {
        double total = 0.0;
        for (const my_torch::Parameter &parameter : network.parameters())
            total += (*parameter.value * *parameter.value).sum();
        return total;
    };
    cr_assert_lt(size(decayed), size(free));
}

Test(weight_decay, rejects_a_negative_decay)
{
    my_torch::Matrix value = my_torch::Matrix::row({1});
    my_torch::Matrix gradient = my_torch::Matrix::row({1});
    my_torch::SGD optimizer;

    optimizer.weight_decay = -0.1;
    cr_assert_throw(optimizer.step({{&value, &gradient}}), std::invalid_argument);
}
