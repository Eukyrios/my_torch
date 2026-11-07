/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of scoring a configuration trained like --train, with early stopping, dropout and L2
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "generator/generator.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "model/layers/layers/dropout/dropout.hpp"
#include "training/hyperparameters/fit_score/fit_score.hpp"

static void add_sample(my_torch::Matrix &inputs, my_torch::Matrix &labels, std::size_t row, double x, double y)
{
    inputs(row, 0) = x;
    inputs(row, 1) = y;
    labels(row, (x > 0) == (y > 0) ? 0 : 1) = 1.0;
}

static void fill_quadrants(my_torch::Matrix &inputs, my_torch::Matrix &labels, unsigned int seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> coordinate(-1.0, 1.0);
    for (std::size_t row = 0; row < inputs.rows(); row++)
        add_sample(inputs, labels, row, coordinate(rng), coordinate(rng));
}

static my_torch::TrainingData quadrants()
{
    my_torch::TrainingData data{my_torch::Matrix(200, 2), my_torch::Matrix(200, 2), my_torch::Matrix(100, 2), my_torch::Matrix(100, 2)};
    fill_quadrants(data.train_inputs, data.train_labels, 1);
    fill_quadrants(data.validation_inputs, data.validation_labels, 2);
    return data;
}

static const my_torch::TrainingData DATA = quadrants();

Test(fit_score, defaults_are_the_retained_configuration)
{
    my_torch::Hyperparameters defaults = my_torch::FitScore::defaults();

    cr_assert_float_eq(defaults.at("learning_rate"), 0.018, 1e-15);
    cr_assert_float_eq(defaults.at("hidden_size"), 47.0, 1e-15);
    cr_assert_float_eq(defaults.at("hidden_layers"), 2.0, 1e-15);
    cr_assert_float_eq(defaults.at("dropout"), 0.0, 1e-15);
    cr_assert_float_eq(defaults.at("weight_decay"), 0.0, 1e-15);
}

Test(fit_score, adds_a_dropout_layer_after_each_hidden_layer_only_when_asked)
{
    std::mt19937 rng(0);
    my_torch::FitScore score(DATA, 5);
    my_torch::Network plain = score.build({{"hidden_size", 8}}, rng);
    my_torch::Network dropped = score.build({{"hidden_size", 8}, {"dropout", 0.3}}, rng);

    cr_assert_eq(plain.layers.size(), 3u);
    cr_assert_eq(dropped.layers.size(), 5u);
    cr_assert_float_eq(dynamic_cast<my_torch::Dropout &>(*dropped.layers[1]).rate, 0.3, 1e-15);
    cr_assert(dynamic_cast<my_torch::DenseLayer &>(*dropped.layers[4]).activation->name == "softmax");
}

Test(fit_score, the_training_has_early_stopping_and_the_l2_asked)
{
    my_torch::FitScore score(DATA, 12, 0, 3);
    my_torch::Training training = score.training({{"weight_decay", 0.001}, {"batch_size", 16}});

    cr_assert_eq(training.max_epochs, 12u);
    cr_assert_eq(training.patience, 3u);
    cr_assert_eq(training.batch_size, 16u);
    cr_assert_float_eq(training.optimizer->weight_decay, 0.001, 1e-15);
}

Test(fit_score, the_score_is_the_validation_error_and_it_learns)
{
    my_torch::FitScore score(DATA, 40);
    my_torch::Hyperparameters configuration = {{"learning_rate", 0.05}, {"hidden_size", 16}, {"batch_size", 8}};
    my_torch::Network network = score.train(configuration);
    double expected = 1.0 - my_torch::accuracy(network(DATA.validation_inputs), DATA.validation_labels);

    cr_assert(score(configuration) == expected);
    cr_assert_lt(expected, 0.3);
}

Test(fit_score, config_builds_the_same_network_and_training_with_the_generator)
{
    my_torch::FitScore score(DATA, 10);
    my_torch::Hyperparameters configuration = {{"hidden_size", 12}, {"hidden_layers", 1}, {"dropout", 0.25}, {"weight_decay", 0.0005}, {"batch_size", 24}};
    my_torch::GeneratedNetwork generated = my_torch::generate_network(score.config(configuration, 50), 0);

    cr_assert_eq(generated.network.layers.size(), 3u);
    cr_assert(generated.network.shape() == my_torch::Shape(2, 2));
    cr_assert_float_eq(dynamic_cast<my_torch::Dropout &>(*generated.network.layers[1]).rate, 0.25, 1e-15);
    cr_assert_eq(generated.training.batch_size, 24u);
    cr_assert_eq(generated.training.max_epochs, 50u);
    cr_assert_float_eq(generated.training.optimizer->weight_decay, 0.0005, 1e-15);
}

Test(fit_score, rejects_unknown_or_invalid_hyperparameters)
{
    my_torch::FitScore score(DATA, 2);

    cr_assert_throw(score({{"activation", 1}}), std::invalid_argument);
    cr_assert_throw(score({{"dropout", 1.0}}), std::invalid_argument);
    cr_assert_throw(score({{"hidden_layers", 0}}), std::invalid_argument);
}
