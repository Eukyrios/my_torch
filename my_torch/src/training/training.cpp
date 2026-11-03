/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A training needs a loss, an optimizer and a batch size of at least 1; each run continues the
** epoch count, so the learning-rate schedule picks up where the last run stopped. The validation
** boards are drawn with a fixed seed, so the same file always keeps the same boards aside, and a
** board never moves from validation to training between two fits
*/

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#include "training/training.hpp"

namespace my_torch {

static const unsigned int VALIDATION_SEED = 0;

static void check_training(const Loss *loss, const Optimizer *optimizer, std::size_t batch_size)
{
    if (!loss || !optimizer)
        throw std::invalid_argument("a training needs a loss and an optimizer");
    if (batch_size < 1)
        throw std::invalid_argument("the batch size must be at least 1, got " + std::to_string(batch_size));
}

static void check_fit(const Training &training, std::size_t n_samples)
{
    if (!(training.validation_split >= 0.0 && training.validation_split < 1.0))
        throw std::invalid_argument("the validation split must be in [0, 1), got " + std::to_string(training.validation_split));
    if (training.max_epochs < 1 || training.patience < 1)
        throw std::invalid_argument("max epochs and patience must be at least 1, got " + std::to_string(training.max_epochs) + " and " + std::to_string(training.patience));
    if (n_samples < 2)
        throw std::invalid_argument("training needs at least 2 boards, got " + std::to_string(n_samples));
}

std::vector<std::size_t> validation_rows(std::size_t n_samples, double validation_split)
{
    std::vector<std::size_t> order(n_samples);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(VALIDATION_SEED);
    std::shuffle(order.begin(), order.end(), rng);
    std::size_t count = std::min(static_cast<std::size_t>(std::round(n_samples * validation_split)), n_samples - 1);
    if (validation_split > 0.0)
        count = std::max<std::size_t>(count, 1);
    order.resize(count);
    std::sort(order.begin(), order.end());
    return order;
}

static std::vector<std::size_t> other_rows(std::size_t n_samples, const std::vector<std::size_t> &rows)
{
    std::vector<std::size_t> others;
    for (std::size_t row = 0, next = 0; row < n_samples; row++) {
        if (next < rows.size() && rows[next] == row)
            next++;
        else
            others.push_back(row);
    }
    return others;
}

static std::vector<Matrix> snapshot(Network &network)
{
    std::vector<Matrix> values;
    for (const Parameter &parameter : network.parameters())
        values.push_back(*parameter.value);
    return values;
}

static void restore(Network &network, const std::vector<Matrix> &values)
{
    std::vector<Parameter> parameters = network.parameters();
    for (std::size_t index = 0; index < parameters.size(); index++)
        *parameters[index].value = values[index];
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

FitResult Training::fit(Network &network, const Matrix &inputs, const Matrix &labels, const std::function<void(const EpochReport &)> &on_epoch)
{
    check_fit(*this, inputs.rows());
    std::vector<std::size_t> validation = validation_rows(inputs.rows(), validation_split);
    std::vector<std::size_t> train = other_rows(inputs.rows(), validation);
    Matrix train_inputs = inputs.select_rows(train);
    Matrix train_labels = labels.select_rows(train);
    Matrix validation_inputs = inputs.select_rows(validation);
    Matrix validation_labels = labels.select_rows(validation);
    // without validation boards nothing can be watched: every epoch counts as the best so far
    auto validation_loss = [&]() { return validation.empty() ? NAN : (*loss)(network(validation_inputs), validation_labels); };
    FitResult result {0, validation_loss(), validation_loss(), 0, false};
    std::vector<Matrix> best = snapshot(network);
    std::size_t without_progress = 0;
    while (result.epochs < max_epochs && without_progress < patience) {
        double epoch_loss = run(network, train_inputs, train_labels, 1)[0];
        double current = validation_loss();
        bool improved = validation.empty() || current < result.best_validation_loss;
        validation_losses.push_back(current);
        result.epochs++;
        without_progress = improved ? 0 : without_progress + 1;
        if (improved) {
            result.best_validation_loss = current;
            result.best_epochs = result.epochs;
            best = snapshot(network);
        }
        if (on_epoch)
            on_epoch(EpochReport {epoch - 1, epoch_loss, current, improved});
    }
    result.stopped_early = without_progress >= patience;
    restore(network, best);
    return result;
}

}
