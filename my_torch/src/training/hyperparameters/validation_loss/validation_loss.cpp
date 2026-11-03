/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Hyperparameters not given take their default; decay multiplies the learning rate every epoch
** (1 keeps it); every score trains from the same seed, so two configurations compare fairly
*/

#include <stdexcept>
#include <string>

#include "model/activations/activations/relu/relu.hpp"
#include "model/activations/activations/softmax/softmax.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "training/hyperparameters/validation_loss/validation_loss.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
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

ValidationLoss::ValidationLoss(const TrainingData &data, std::size_t epochs, unsigned int seed)
:
    data(data), epochs(epochs), seed(seed)
{
    check_data(data);
}

Hyperparameters ValidationLoss::defaults()
{
    return {{"learning_rate", 0.01}, {"momentum", 0.9}, {"batch_size", 32}, {"hidden_size", 64}, {"hidden_layers", 1}, {"decay", 1.0}};
}

Network ValidationLoss::build(const Hyperparameters &hyperparameters, std::mt19937 &rng) const
{
    Hyperparameters values = with_defaults(hyperparameters);
    std::size_t width = whole(values, "hidden_size");
    std::size_t inputs = data.train_inputs.cols();
    std::vector<std::unique_ptr<Layer>> layers;
    for (std::size_t layer = 0; layer < whole(values, "hidden_layers"); layer++) {
        layers.push_back(DenseLayer::create(inputs, width, std::make_unique<ReLU>(), rng));
        inputs = width;
    }
    layers.push_back(DenseLayer::create(inputs, data.train_labels.cols(), std::make_unique<Softmax>(), rng));
    return Network(std::move(layers));
}

Network ValidationLoss::train(const Hyperparameters &hyperparameters) const
{
    Hyperparameters values = with_defaults(hyperparameters);
    std::mt19937 rng(seed);
    Network network = build(values, rng);
    Momentum optimizer(values.at("learning_rate"), values.at("momentum"), std::make_unique<ExponentialDecay>(values.at("decay")));
    network.train(data.train_inputs, data.train_labels, CrossEntropy(), optimizer, epochs, whole(values, "batch_size"), rng);
    return network;
}

double ValidationLoss::operator()(const Hyperparameters &hyperparameters) const
{
    Network network = train(hyperparameters);
    return CrossEntropy()(network(data.validation_inputs), data.validation_labels);
}

}
