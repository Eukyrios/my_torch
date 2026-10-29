/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of training a network with shuffled mini-batches
*/

#include <algorithm>
#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "model/activations/activations/softmax/softmax.hpp"
#include "model/activations/activations/tanh/tanh.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "model/network/network.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/optimizers/optimizers/sgd/sgd.hpp"
#include "training/schedules/schedules/step_decay/step_decay.hpp"
#include "training/schedules/schedules/time_decay/time_decay.hpp"

static my_torch::Matrix training_inputs()
{
    std::mt19937 rng(3);
    std::normal_distribution<double> normal(0.0, 1.0);
    my_torch::Matrix inputs(20, 4);
    for (std::size_t row = 0; row < 20; row++)
        for (std::size_t col = 0; col < 4; col++)
            inputs(row, col) = normal(rng);
    return inputs;
}

static my_torch::Matrix training_labels()
{
    my_torch::Matrix labels(20, 3);
    for (std::size_t row = 0; row < 20; row++)
        labels(row, row % 3) = 1.0;
    return labels;
}

static const my_torch::Matrix INPUTS = training_inputs();
static const my_torch::Matrix LABELS = training_labels();

static my_torch::Network small_network(unsigned int seed = 0)
{
    std::mt19937 rng(seed);
    std::vector<std::unique_ptr<my_torch::Layer>> layers;
    layers.push_back(my_torch::DenseLayer::create(4, 8, std::make_unique<my_torch::Tanh>(), rng));
    layers.push_back(my_torch::DenseLayer::create(8, 3, std::make_unique<my_torch::Softmax>(), rng));
    return my_torch::Network(std::move(layers));
}

Test(training, train_returns_one_mean_loss_per_epoch)
{
    std::mt19937 rng(0);
    my_torch::SGD optimizer(0.1);
    std::vector<double> losses = small_network().train(INPUTS, LABELS, my_torch::CrossEntropy(), optimizer, 5, 4, rng);

    cr_assert_eq(losses.size(), 5u);
    for (double loss : losses)
        cr_assert(std::isfinite(loss));
}

Test(training, the_loss_goes_down_while_training)
{
    std::mt19937 rng(0);
    my_torch::SGD optimizer(0.1);
    std::vector<double> losses = small_network().train(INPUTS, LABELS, my_torch::CrossEntropy(), optimizer, 200, 4, rng);

    cr_assert_lt(losses.back(), losses.front() / 2);
}

Test(training, the_first_epoch_loss_is_the_mean_over_all_samples_with_one_full_batch)
{
    std::mt19937 rng(0);
    my_torch::SGD optimizer(0.1);
    my_torch::Network network = small_network();
    double expected = my_torch::CrossEntropy()(network(INPUTS), LABELS);
    std::vector<double> losses = network.train(INPUTS, LABELS, my_torch::CrossEntropy(), optimizer, 1, 20, rng);

    cr_assert_float_eq(losses[0], expected, 1e-12);
}

Test(training, training_is_reproducible_with_the_same_seeds)
{
    std::mt19937 first_rng(5);
    std::mt19937 second_rng(5);
    my_torch::SGD first_optimizer(0.1);
    my_torch::SGD second_optimizer(0.1);
    std::vector<double> first = small_network().train(INPUTS, LABELS, my_torch::CrossEntropy(), first_optimizer, 10, 4, first_rng);
    std::vector<double> second = small_network().train(INPUTS, LABELS, my_torch::CrossEntropy(), second_optimizer, 10, 4, second_rng);

    cr_assert(first == second);
}

Test(training, any_batch_size_trains_on_every_sample)
{
    for (std::size_t batch_size : {1, 3, 7, 20, 64}) {
        std::mt19937 rng(0);
        my_torch::SGD optimizer(0.05);
        cr_assert_eq(small_network().train(INPUTS, LABELS, my_torch::CrossEntropy(), optimizer, 2, batch_size, rng).size(), 2u);
    }
}

Test(training, shuffled_batches_cover_every_sample_once)
{
    std::mt19937 rng(0);
    std::vector<std::vector<std::size_t>> batches = my_torch::shuffled_batches(10, 3, rng);
    std::vector<std::size_t> every;

    cr_assert_eq(batches.size(), 4u);
    cr_assert_eq(batches[3].size(), 1u);
    for (const std::vector<std::size_t> &batch : batches)
        every.insert(every.end(), batch.begin(), batch.end());
    std::sort(every.begin(), every.end());
    for (std::size_t index = 0; index < 10; index++)
        cr_assert_eq(every[index], index);
}

Test(training, shuffled_batches_change_order_between_calls)
{
    std::mt19937 rng(0);

    cr_assert(my_torch::shuffled_batches(10, 10, rng) != my_torch::shuffled_batches(10, 10, rng));
}

Test(training, rejects_a_different_number_of_samples_and_labels)
{
    std::mt19937 rng(0);
    my_torch::SGD optimizer;

    cr_assert_throw(small_network().train(INPUTS, LABELS.select_rows({0, 1}), my_torch::CrossEntropy(), optimizer, 1, 4, rng), std::invalid_argument);
}

Test(training, rejects_epochs_or_batch_size_below_one)
{
    std::mt19937 rng(0);
    my_torch::SGD optimizer;

    cr_assert_throw(small_network().train(INPUTS, LABELS, my_torch::CrossEntropy(), optimizer, 0, 4, rng), std::invalid_argument);
    cr_assert_throw(small_network().train(INPUTS, LABELS, my_torch::CrossEntropy(), optimizer, 1, 0, rng), std::invalid_argument);
}

Test(training, the_optimizer_follows_its_schedule_over_the_epochs)
{
    std::mt19937 rng(0);
    my_torch::SGD optimizer(0.1, std::make_unique<my_torch::StepDecay>(0.5, 1));

    small_network().train(INPUTS, LABELS, my_torch::CrossEntropy(), optimizer, 3, 4, rng);
    cr_assert_float_eq(optimizer.learning_rate, 0.025, 1e-15);
}

Test(training, a_decaying_rate_still_lowers_the_loss)
{
    std::mt19937 rng(0);
    my_torch::Momentum optimizer(0.05, 0.9, std::make_unique<my_torch::TimeDecay>(0.01));
    std::vector<double> losses = small_network().train(INPUTS, LABELS, my_torch::CrossEntropy(), optimizer, 100, 4, rng);

    cr_assert_lt(losses.back(), losses.front() / 2);
}
