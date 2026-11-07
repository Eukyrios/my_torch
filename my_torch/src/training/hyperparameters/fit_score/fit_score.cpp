/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Missing hyperparameters keep the retained configuration (2 x 47, learning rate 0.018, ...); a dropout
** layer follows every hidden layer when dropout > 0, and weight_decay is the optimizer's L2
*/

#include <cmath>
#include <stdexcept>
#include <string>

#include "model/activations/activations/relu/relu.hpp"
#include "model/activations/activations/softmax/softmax.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "model/layers/layers/dropout/dropout.hpp"
#include "training/hyperparameters/fit_score/fit_score.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/schedules/schedules/exponential_decay/exponential_decay.hpp"

namespace my_torch {

static void check_data(const TrainingData &data)
{
    if (data.train_inputs.rows() == 0 || data.train_inputs.rows() != data.train_labels.rows() || data.validation_inputs.rows() == 0 || data.validation_inputs.rows() != data.validation_labels.rows())
        throw std::invalid_argument("the training and validation data need samples and one label per sample");
    if (data.train_inputs.cols() != data.validation_inputs.cols() || data.train_labels.cols() != data.validation_labels.cols())
        throw std::invalid_argument("the training and validation data need the same number of inputs and outputs");
}

static Hyperparameters with_defaults(const Hyperparameters &given)
{
    Hyperparameters values = FitScore::defaults();
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
    return static_cast<std::size_t>(std::lround(value));
}

static double dropout_rate(const Hyperparameters &values)
{
    double rate = values.at("dropout");
    if (!(rate >= 0.0 && rate < 1.0))
        throw std::invalid_argument("hyperparameter 'dropout' must be in [0, 1), got " + std::to_string(rate));
    return rate;
}

static Json dense_json(std::size_t outputs, const char *activation, const char *initialization)
{
    return Json(Json::Object {{"type", "dense"}, {"outputs", outputs}, {"activation", activation}, {"initialization", initialization}});
}

static Json layers_json(const Hyperparameters &values, std::size_t outputs)
{
    Json::Array layers;
    for (std::size_t layer = 0; layer < whole(values, "hidden_layers"); layer++) {
        layers.push_back(dense_json(whole(values, "hidden_size"), "relu", "he"));
        if (dropout_rate(values) > 0.0)
            layers.push_back(Json(Json::Object {{"type", "dropout"}, {"rate", dropout_rate(values)}}));
    }
    layers.push_back(dense_json(outputs, "softmax", "xavier"));
    return Json(layers);
}

FitScore::FitScore(const TrainingData &data, std::size_t max_epochs, unsigned int seed, std::size_t patience)
:
    data(data), max_epochs(max_epochs), seed(seed), patience(patience)
{
    check_data(data);
}

Hyperparameters FitScore::defaults()
{
    return {{"learning_rate", 0.018}, {"momentum", 0.53}, {"batch_size", 9}, {"hidden_size", 47}, {"hidden_layers", 2}, {"decay", 0.86}, {"dropout", 0.0}, {"weight_decay", 0.0}};
}

Network FitScore::build(const Hyperparameters &hyperparameters, std::mt19937 &rng) const
{
    Hyperparameters values = with_defaults(hyperparameters);
    std::size_t inputs = data.train_inputs.cols();
    std::vector<std::unique_ptr<Layer>> layers;
    for (std::size_t layer = 0; layer < whole(values, "hidden_layers"); layer++) {
        layers.push_back(DenseLayer::create(inputs, whole(values, "hidden_size"), std::make_unique<ReLU>(), rng));
        if (dropout_rate(values) > 0.0)
            layers.push_back(std::make_unique<Dropout>(dropout_rate(values), std::mt19937(rng())));
        inputs = whole(values, "hidden_size");
    }
    layers.push_back(DenseLayer::create(inputs, data.train_labels.cols(), std::make_unique<Softmax>(), rng));
    return Network(std::move(layers));
}

Training FitScore::training(const Hyperparameters &hyperparameters) const
{
    Hyperparameters values = with_defaults(hyperparameters);
    std::unique_ptr<Optimizer> optimizer = std::make_unique<Momentum>(values.at("learning_rate"), values.at("momentum"), std::make_unique<ExponentialDecay>(values.at("decay")));
    optimizer->weight_decay = values.at("weight_decay");
    Training training(std::make_unique<CrossEntropy>(), std::move(optimizer), whole(values, "batch_size"), std::mt19937(seed));
    training.patience = patience;
    training.max_epochs = max_epochs;
    return training;
}

Network FitScore::train(const Hyperparameters &hyperparameters) const
{
    std::mt19937 rng(seed);
    Network network = build(hyperparameters, rng);
    Training fit = training(hyperparameters);
    fit.fit(network, data.train_inputs, data.train_labels);
    return network;
}

double FitScore::operator()(const Hyperparameters &hyperparameters) const
{
    Network network = train(hyperparameters);
    return 1.0 - accuracy(network(data.validation_inputs), data.validation_labels);
}

Json FitScore::config(const Hyperparameters &hyperparameters, std::size_t epochs) const
{
    Hyperparameters values = with_defaults(hyperparameters);
    Json optimizer(Json::Object {
        {"name", "momentum"}, {"learning_rate", values.at("learning_rate")}, {"momentum", values.at("momentum")},
        {"weight_decay", values.at("weight_decay")}, {"schedule", Json(Json::Object {{"name", "exponential_decay"}, {"factor", values.at("decay")}})},
    });
    Json training(Json::Object {
        {"batch_size", whole(values, "batch_size")}, {"loss", Json(Json::Object {{"name", "cross_entropy"}})}, {"optimizer", optimizer},
        {"validation_split", 0.1}, {"patience", patience}, {"max_epochs", epochs},
    });
    return Json(Json::Object {{"inputs", data.train_inputs.cols()}, {"seed", 0}, {"layers", layers_json(values, data.train_labels.cols())}, {"training", training}});
}

}
