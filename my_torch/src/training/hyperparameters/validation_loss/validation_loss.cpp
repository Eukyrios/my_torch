/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Hyperparameters not given take their default (relu, cross-entropy); activation and loss are indices
** into their choices; every score trains from the same seed, so two configurations compare fairly
*/

#include <stdexcept>
#include <string>

#include "model/activations/activation.hpp"
#include "model/activations/activations/softmax/softmax.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "training/hyperparameters/validation_loss/validation_loss.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/losses/losses/mean_squared_error/mean_squared_error.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/schedules/schedules/exponential_decay/exponential_decay.hpp"

namespace my_torch {

static void check_same_rows(const Matrix &inputs, const Matrix &labels, const char *part)
{
    if (inputs.rows() == 0 || inputs.rows() != labels.rows())
        throw std::invalid_argument(std::string("the ") + part + " data needs at least one sample and one label per sample");
}

static void check_data(const TrainingData &data)
{
    check_same_rows(data.train_inputs, data.train_labels, "training");
    check_same_rows(data.validation_inputs, data.validation_labels, "validation");
    if (data.train_inputs.cols() != data.validation_inputs.cols() || data.train_labels.cols() != data.validation_labels.cols())
        throw std::invalid_argument("the training and validation data need the same number of inputs and outputs");
}

static Hyperparameters with_defaults(const Hyperparameters &given)
{
    Hyperparameters values = ValidationLoss::defaults();
    for (const auto &[name, value] : given) {
        if (!values.count(name))
            throw std::invalid_argument("unknown hyperparameter '" + name + "'");
        values[name] = value;
    }
    return values;
}

static std::size_t whole(const Hyperparameters &values, const std::string &name)
{
    double value = values.at(name);
    if (value < 1.0)
        throw std::invalid_argument("hyperparameter '" + name + "' must be at least 1, got " + std::to_string(value));
    return static_cast<std::size_t>(value + 0.5);
}

static const std::string &chosen(const std::vector<std::string> &choices, const Hyperparameters &values, const std::string &name)
{
    double index = values.at(name);
    if (!(index >= 0.0 && index < static_cast<double>(choices.size())))
        throw std::invalid_argument("hyperparameter '" + name + "' must be an index below " + std::to_string(choices.size()) + ", got " + std::to_string(index));
    return choices[static_cast<std::size_t>(index)];
}

ValidationLoss::ValidationLoss(const TrainingData &data, std::size_t epochs, unsigned int seed)
:
    data(data), epochs(epochs), seed(seed)
{
    check_data(data);
}

Hyperparameters ValidationLoss::defaults()
{
    return {{"learning_rate", 0.01}, {"momentum", 0.9}, {"batch_size", 32}, {"hidden_size", 64}, {"hidden_layers", 1}, {"decay", 1.0}, {"activation", 0}, {"loss", 0}};
}

const std::vector<std::string> &ValidationLoss::activation_choices()
{
    static const std::vector<std::string> choices = {"relu", "tanh"};
    return choices;
}

const std::vector<std::string> &ValidationLoss::loss_choices()
{
    static const std::vector<std::string> choices = {"cross_entropy", "mean_squared_error"};
    return choices;
}

std::string ValidationLoss::activation_name(const Hyperparameters &hyperparameters)
{
    return chosen(activation_choices(), with_defaults(hyperparameters), "activation");
}

std::string ValidationLoss::loss_name(const Hyperparameters &hyperparameters)
{
    return chosen(loss_choices(), with_defaults(hyperparameters), "loss");
}

Network ValidationLoss::build(const Hyperparameters &hyperparameters, std::mt19937 &rng) const
{
    Hyperparameters values = with_defaults(hyperparameters);
    std::size_t width = whole(values, "hidden_size");
    std::size_t inputs = data.train_inputs.cols();
    std::vector<std::unique_ptr<Layer>> layers;
    for (std::size_t layer = 0; layer < whole(values, "hidden_layers"); layer++) {
        layers.push_back(DenseLayer::create(inputs, width, Activation::from_name(activation_name(values)), rng));
        inputs = width;
    }
    layers.push_back(DenseLayer::create(inputs, data.train_labels.cols(), std::make_unique<Softmax>(), rng));
    return Network(std::move(layers));
}

std::unique_ptr<Loss> ValidationLoss::loss(const Hyperparameters &hyperparameters) const
{
    if (loss_name(hyperparameters) == "mean_squared_error")
        return std::make_unique<MeanSquaredError>();
    return std::make_unique<CrossEntropy>();
}

std::unique_ptr<Optimizer> ValidationLoss::optimizer(const Hyperparameters &hyperparameters) const
{
    Hyperparameters values = with_defaults(hyperparameters);
    return std::make_unique<Momentum>(values.at("learning_rate"), values.at("momentum"), std::make_unique<ExponentialDecay>(values.at("decay")));
}

std::size_t ValidationLoss::batch_size(const Hyperparameters &hyperparameters) const
{
    return whole(with_defaults(hyperparameters), "batch_size");
}

Network ValidationLoss::train(const Hyperparameters &hyperparameters) const
{
    std::mt19937 rng(seed);
    Network network = build(hyperparameters, rng);
    std::unique_ptr<Optimizer> chosen_optimizer = optimizer(hyperparameters);
    network.train(data.train_inputs, data.train_labels, *loss(hyperparameters), *chosen_optimizer, epochs, batch_size(hyperparameters), rng);
    return network;
}

double ValidationLoss::operator()(const Hyperparameters &hyperparameters) const
{
    Network network = train(hyperparameters);
    return CrossEntropy()(network(data.validation_inputs), data.validation_labels);
}

double ValidationLoss::error(const Hyperparameters &hyperparameters) const
{
    Network network = train(hyperparameters);
    return 1.0 - accuracy(network(data.validation_inputs), data.validation_labels);
}

ValidationError::ValidationError(const ValidationLoss &score)
:
    _score(&score)
{}

double ValidationError::operator()(const Hyperparameters &hyperparameters) const
{
    return _score->error(hyperparameters);
}

double accuracy(const Matrix &predictions, const Matrix &labels)
{
    std::vector<std::size_t> predicted = predictions.row_argmax();
    std::vector<std::size_t> expected = labels.row_argmax();
    std::size_t right = 0;
    for (std::size_t row = 0; row < predicted.size(); row++)
        right += predicted[row] == expected[row];
    return predicted.empty() ? 0.0 : static_cast<double>(right) / predicted.size();
}

}
