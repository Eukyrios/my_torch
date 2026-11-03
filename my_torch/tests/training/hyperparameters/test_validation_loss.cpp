/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of scoring configurations by their validation loss, alone and inside a particle swarm
*/

#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "training/hyperparameters/algorithms/particle_swarm/particle_swarm.hpp"
#include "training/hyperparameters/validation_loss/validation_loss.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"

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
    my_torch::TrainingData data{my_torch::Matrix(80, 2), my_torch::Matrix(80, 2), my_torch::Matrix(40, 2), my_torch::Matrix(40, 2)};
    fill_quadrants(data.train_inputs, data.train_labels, 1);
    fill_quadrants(data.validation_inputs, data.validation_labels, 2);
    return data;
}

static const my_torch::TrainingData DATA = quadrants();

Test(validation_loss, a_good_configuration_scores_better_than_a_learning_rate_too_small_to_learn)
{
    my_torch::ValidationLoss score(DATA, 40);

    cr_assert_lt(score({{"learning_rate", 0.05}, {"hidden_size", 16}, {"batch_size", 8}}), score({{"learning_rate", 1e-6}, {"hidden_size", 16}, {"batch_size", 8}}));
}

Test(validation_loss, a_sensible_configuration_does_clearly_better_than_guessing)
{
    my_torch::ValidationLoss score(DATA, 60);
    double guessing = std::log(2.0);

    cr_assert_lt(score({{"learning_rate", 0.05}, {"hidden_size", 16}, {"batch_size", 8}}), 0.75 * guessing);
}

Test(validation_loss, the_same_configuration_always_gets_the_same_score)
{
    my_torch::ValidationLoss score(DATA, 10, 3);
    my_torch::Hyperparameters configuration = {{"learning_rate", 0.02}, {"hidden_size", 8}};

    cr_assert(score(configuration) == score(configuration));
}

Test(validation_loss, builds_the_requested_layers)
{
    std::mt19937 rng(0);
    my_torch::Network network = my_torch::ValidationLoss(DATA, 1).build({{"hidden_layers", 3}, {"hidden_size", 12}}, rng);

    cr_assert_eq(network.layers.size(), 4u);
    cr_assert(network.layers[1]->shape() == my_torch::Shape(12, 12));
    cr_assert(network.shape() == my_torch::Shape(2, 2));
}

Test(validation_loss, the_score_is_the_validation_loss_of_the_trained_network)
{
    my_torch::ValidationLoss score(DATA, 10);
    my_torch::Hyperparameters configuration = {{"learning_rate", 0.05}, {"hidden_size", 8}};
    my_torch::Network network = score.train(configuration);
    my_torch::Matrix predictions = network(DATA.validation_inputs);

    cr_assert(score(configuration) == my_torch::CrossEntropy()(predictions, DATA.validation_labels));
}

Test(validation_loss, missing_hyperparameters_take_their_default)
{
    my_torch::ValidationLoss score(DATA, 5);

    cr_assert(score({}) == score(my_torch::ValidationLoss::defaults()));
}

Test(validation_loss, rejects_unknown_or_invalid_hyperparameters)
{
    my_torch::ValidationLoss score(DATA, 1);

    cr_assert_throw(score({{"learnig_rate", 0.1}}), std::invalid_argument);
    cr_assert_throw(score({{"batch_size", 0}}), std::invalid_argument);
    cr_assert_throw(score({{"hidden_layers", 0}}), std::invalid_argument);
}

Test(validation_loss, rejects_inconsistent_data)
{
    my_torch::TrainingData few_labels{my_torch::Matrix(4, 2), my_torch::Matrix(3, 2), my_torch::Matrix(2, 2), my_torch::Matrix(2, 2)};
    my_torch::TrainingData other_inputs{my_torch::Matrix(4, 2), my_torch::Matrix(4, 2), my_torch::Matrix(2, 3), my_torch::Matrix(2, 2)};

    cr_assert_throw(my_torch::ValidationLoss(few_labels, 1), std::invalid_argument);
    cr_assert_throw(my_torch::ValidationLoss(other_inputs, 1), std::invalid_argument);
}

Test(validation_loss, a_particle_swarm_improves_on_its_starting_configurations)
{
    std::mt19937 rng(0);
    my_torch::SearchSpace space({
        {"learning_rate", 1e-5, 0.3, true, false},
        {"hidden_size", 2, 32, true, true},
    });
    my_torch::SwarmSettings settings;
    settings.particles = 4;
    settings.iterations = 4;
    my_torch::SearchResult result = my_torch::ParticleSwarm(space, settings).minimize(my_torch::ValidationLoss(DATA, 20), rng);

    cr_assert(std::isfinite(result.best_score));
    cr_assert_lt(result.best_per_iteration.back(), 0.5);
    cr_assert_leq(result.best_per_iteration.back(), result.best_per_iteration.front());
}
