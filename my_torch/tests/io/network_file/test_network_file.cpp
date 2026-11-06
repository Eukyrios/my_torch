/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of saving and loading networks: what a round trip keeps, resuming a training, and
** every kind of broken file being refused with NetworkFileError
*/

#include <bit>
#include <cstdint>
#include <criterion/criterion.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

#include "io/network_file/network_file.hpp"
#include "io/npy/npy.hpp"
#include "io/zip/zip.hpp"
#include "model/activations/activations/relu/relu.hpp"
#include "model/activations/activations/softmax/softmax.hpp"
#include "model/activations/activations/tanh/tanh.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "model/layers/layers/dropout/dropout.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/losses/losses/mean_squared_error/mean_squared_error.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/optimizers/optimizers/sgd/sgd.hpp"
#include "training/schedules/schedules/exponential_decay/exponential_decay.hpp"
#include "training/schedules/schedules/step_decay/step_decay.hpp"
#include "training/schedules/schedules/time_decay/time_decay.hpp"

using my_torch::Json;
using my_torch::Matrix;
using my_torch::NetworkFileError;

static const Matrix INPUTS({{0, 0, 1}, {0, 1, 0}, {1, 0, 0}, {1, 1, 1}, {0, 1, 1}, {1, 0, 1}});
static const Matrix LABELS({{1, 0}, {0, 1}, {0, 1}, {1, 0}, {1, 0}, {0, 1}});

static std::string temporary_path(const std::string &name)
{
    std::filesystem::path directory = std::filesystem::temp_directory_path() / ("my_torch_tests_" + std::to_string(getpid()));
    std::filesystem::create_directories(directory);
    return (directory / name).string();
}

static void write_bytes(const std::string &path, const std::string &bytes)
{
    std::ofstream(path, std::ios::binary) << bytes;
}

static std::string read_bytes(const std::string &path)
{
    std::ostringstream bytes;
    bytes << std::ifstream(path, std::ios::binary).rdbuf();
    return bytes.str();
}

static my_torch::Network small_network(unsigned int seed = 1)
{
    std::mt19937 rng(seed);
    std::vector<std::unique_ptr<my_torch::Layer>> layers;
    layers.push_back(my_torch::DenseLayer::create(3, 5, std::make_unique<my_torch::ReLU>(), rng));
    layers.push_back(my_torch::DenseLayer::create(5, 4, std::make_unique<my_torch::Tanh>(), rng));
    layers.push_back(my_torch::DenseLayer::create(4, 2, std::make_unique<my_torch::Softmax>(), rng));
    return my_torch::Network(std::move(layers));
}

static my_torch::Training momentum_training()
{
    return my_torch::Training(std::make_unique<my_torch::CrossEntropy>(1e-9), std::make_unique<my_torch::Momentum>(0.05, 0.8, std::make_unique<my_torch::StepDecay>(0.5, 4)), 2, std::mt19937(11));
}

static const my_torch::DenseLayer &dense(const my_torch::Network &network, std::size_t layer)
{
    return dynamic_cast<const my_torch::DenseLayer &>(*network.layers[layer]);
}

static void assert_same_bits(const Matrix &actual, const Matrix &expected)
{
    cr_assert_eq(actual.rows(), expected.rows());
    cr_assert_eq(actual.cols(), expected.cols());
    for (std::size_t row = 0; row < expected.rows(); row++)
        for (std::size_t col = 0; col < expected.cols(); col++)
            cr_assert_eq(std::bit_cast<std::uint64_t>(actual(row, col)), std::bit_cast<std::uint64_t>(expected(row, col)));
}

static void assert_same_network(const my_torch::Network &actual, const my_torch::Network &expected)
{
    cr_assert_eq(actual.layers.size(), expected.layers.size());
    for (std::size_t layer = 0; layer < expected.layers.size(); layer++) {
        cr_assert_str_eq(actual.layers[layer]->name.c_str(), expected.layers[layer]->name.c_str());
        cr_assert_str_eq(dense(actual, layer).activation->name.c_str(), dense(expected, layer).activation->name.c_str());
        assert_same_bits(dense(actual, layer).weights, dense(expected, layer).weights);
        assert_same_bits(dense(actual, layer).biases, dense(expected, layer).biases);
    }
}

// a .nn file made by hand, to break one part of it
static std::string handmade(const std::string &header, const my_torch::ZipEntries &arrays = {})
{
    my_torch::ZipEntries entries = {{"header.json", header}};
    entries.insert(entries.end(), arrays.begin(), arrays.end());
    std::string path = temporary_path("handmade.nn");
    write_bytes(path, my_torch::to_zip(entries));
    return path;
}

static const std::string ONE_LAYER = "\"layers\": [{\"type\": \"dense\", \"activation\": \"relu\", \"shape\": [2, 1]}]";

static my_torch::ZipEntries one_layer_arrays()
{
    return {{"layer_0_weights.npy", my_torch::to_npy(Matrix({{1}, {2}}))}, {"layer_0_biases.npy", my_torch::to_npy(Matrix::row({3}), true)}};
}

static void assert_refused(const std::string &path, const std::string &expected_message)
{
    try {
        my_torch::load_network(path);
        cr_assert_fail("loading %s should fail with '%s'", path.c_str(), expected_message.c_str());
    } catch (const NetworkFileError &error) {
        cr_assert_neq(std::string(error.what()).find(expected_message), std::string::npos, "'%s' does not say '%s'", error.what(), expected_message.c_str());
    }
}

/* Round trip */

Test(network_file, a_round_trip_keeps_layers_activations_weights_and_biases_bit_for_bit)
{
    std::string path = temporary_path("round_trip.nn");
    my_torch::Network network = small_network();

    my_torch::save_network(path, network);
    my_torch::SavedNetwork loaded = my_torch::load_network(path);
    assert_same_network(loaded.network, network);
    assert_same_bits(loaded.network(INPUTS), network(INPUTS));
}

Test(network_file, saving_a_loaded_network_gives_the_same_file)
{
    std::string path = temporary_path("first.nn");
    std::string again = temporary_path("again.nn");
    my_torch::Network network = small_network();

    my_torch::save_network(path, network, Json::Object {{"dataset", "data/train.txt"}});
    my_torch::SavedNetwork loaded = my_torch::load_network(path);
    my_torch::save_network(again, loaded.network, loaded.metadata);
    cr_assert(read_bytes(path) == read_bytes(again));
}

Test(network_file, keeps_the_metadata_and_has_none_by_default)
{
    std::string path = temporary_path("metadata.nn");
    my_torch::Network network = small_network();

    my_torch::save_network(path, network, Json::Object {{"dataset", "data/train.txt"}, {"accuracy", 0.97}});
    my_torch::SavedNetwork loaded = my_torch::load_network(path);
    cr_assert_str_eq(loaded.metadata.find("dataset")->as_string().c_str(), "data/train.txt");
    cr_assert_eq(loaded.metadata.find("accuracy")->as_number(), 0.97);
    cr_assert_not(loaded.training.has_value());
    my_torch::save_network(path, network);
    cr_assert(my_torch::load_network(path).metadata.as_object().empty());
}

Test(network_file, the_header_is_readable_json_with_the_learning_rate)
{
    std::string path = temporary_path("readable.nn");
    my_torch::Network network = small_network();
    my_torch::Training training = momentum_training();

    training.run(network, INPUTS, LABELS, 5);
    my_torch::save_network(path, network, Json::Object(), &training);
    my_torch::ZipEntries entries = my_torch::from_zip(read_bytes(path));
    Json header = Json::parse(entries[0].second);
    cr_assert_str_eq(entries[0].first.c_str(), "header.json");
    cr_assert_str_eq(header.find("format")->as_string().c_str(), "my_torch");
    cr_assert_eq(header.find("version")->as_number(), 1);
    cr_assert_neq(entries[0].second.find("\"learning_rate\": 0.05"), std::string::npos);
    cr_assert_neq(entries[0].second.find("\"current_learning_rate\": 0.025"), std::string::npos);
    cr_assert_neq(entries[0].second.find("\"shape\": [3, 5]"), std::string::npos);
    cr_assert_str_eq(entries[1].first.c_str(), "layer_0_weights.npy");
    cr_assert_str_eq(entries[2].first.c_str(), "layer_0_biases.npy");
    cr_assert_str_eq(entries.back().first.c_str(), "optimizer_5.npy");
}

/* Training */

Test(network_file, keeps_the_whole_training)
{
    std::string path = temporary_path("training.nn");
    my_torch::Network network = small_network();
    my_torch::Training training = momentum_training();

    training.run(network, INPUTS, LABELS, 5);
    my_torch::save_network(path, network, Json::Object(), &training);
    my_torch::SavedNetwork loaded = my_torch::load_network(path);
    cr_assert(loaded.training.has_value());
    my_torch::Training &read = *loaded.training;
    cr_assert_eq(read.epoch, 5);
    cr_assert(read.losses == training.losses);
    cr_assert_eq(read.batch_size, 2);
    cr_assert(read.rng == training.rng);
    cr_assert_str_eq(read.loss->name.c_str(), "cross_entropy");
    cr_assert_eq(dynamic_cast<my_torch::CrossEntropy &>(*read.loss).epsilon, 1e-9);
    my_torch::Momentum &momentum = dynamic_cast<my_torch::Momentum &>(*read.optimizer);
    cr_assert_eq(momentum.initial_learning_rate, 0.05);
    cr_assert_eq(momentum.learning_rate, training.optimizer->learning_rate);
    cr_assert_eq(momentum.momentum, 0.8);
    my_torch::StepDecay &schedule = dynamic_cast<my_torch::StepDecay &>(*momentum.schedule);
    cr_assert_eq(schedule.factor, 0.5);
    cr_assert_eq(schedule.every, 4);
    cr_assert_eq(momentum.velocities.size(), 6);
}

Test(network_file, a_resumed_training_gives_the_same_network_as_one_in_one_go)
{
    std::string path = temporary_path("resume.nn");
    my_torch::Network in_one_go = small_network();
    my_torch::Network first_half = small_network();
    my_torch::Training one_go = momentum_training();
    my_torch::Training first = momentum_training();

    one_go.run(in_one_go, INPUTS, LABELS, 10);
    first.run(first_half, INPUTS, LABELS, 5);
    my_torch::save_network(path, first_half, Json::Object(), &first);
    my_torch::SavedNetwork resumed = my_torch::load_network(path);
    resumed.training->run(resumed.network, INPUTS, LABELS, 5);
    assert_same_network(resumed.network, in_one_go);
    cr_assert(resumed.training->losses == one_go.losses);
    cr_assert_eq(resumed.training->epoch, 10);
}

Test(network_file, keeps_every_loss_optimizer_and_schedule)
{
    std::string path = temporary_path("settings.nn");
    my_torch::Network network = small_network();
    my_torch::Training time_decay(std::make_unique<my_torch::MeanSquaredError>(), std::make_unique<my_torch::SGD>(0.3, std::make_unique<my_torch::TimeDecay>(0.25)));
    my_torch::Training exponential(std::make_unique<my_torch::CrossEntropy>(), std::make_unique<my_torch::SGD>(0.2, std::make_unique<my_torch::ExponentialDecay>(0.9)));
    my_torch::Training constant(std::make_unique<my_torch::CrossEntropy>(), std::make_unique<my_torch::SGD>(0.1));

    my_torch::save_network(path, network, Json::Object(), &time_decay);
    my_torch::SavedNetwork loaded = my_torch::load_network(path);
    cr_assert_str_eq(loaded.training->loss->name.c_str(), "mean_squared_error");
    cr_assert_str_eq(loaded.training->optimizer->name.c_str(), "sgd");
    cr_assert_eq(dynamic_cast<my_torch::TimeDecay &>(*loaded.training->optimizer->schedule).decay, 0.25);
    my_torch::save_network(path, network, Json::Object(), &exponential);
    cr_assert_eq(dynamic_cast<my_torch::ExponentialDecay &>(*my_torch::load_network(path).training->optimizer->schedule).factor, 0.9);
    my_torch::save_network(path, network, Json::Object(), &constant);
    loaded = my_torch::load_network(path);
    cr_assert_str_eq(loaded.training->optimizer->schedule->name.c_str(), "constant");
}

/* Saving errors */

Test(network_file, saving_where_it_cannot_write_fails_and_keeps_nothing)
{
    my_torch::Network network = small_network();

    cr_assert_throw(my_torch::save_network("/no/such/directory/net.nn", network), NetworkFileError);
    cr_assert_throw(my_torch::save_network(temporary_path("bad_metadata.nn"), network, Json(1)), NetworkFileError);
    cr_assert_not(std::filesystem::exists(temporary_path("bad_metadata.nn")));
}

/* Loading errors */

Test(network_file, refuses_a_missing_or_unreadable_file)
{
    assert_refused(temporary_path("does_not_exist.nn"), "cannot open the network file");
    assert_refused(std::filesystem::temp_directory_path().string(), "cannot open the network file");
}

Test(network_file, refuses_a_file_that_is_not_a_network)
{
    std::string path = temporary_path("not_a_network.nn");

    write_bytes(path, "");
    assert_refused(path, "not a zip archive");
    write_bytes(path, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n");
    assert_refused(path, "not a zip archive");
    write_bytes(path, my_torch::to_zip({{"other.txt", "hello"}}));
    assert_refused(path, "no header.json");
    assert_refused(handmade("{\"format\": \"keras\", \"version\": 1}"), "not a my_torch network");
    assert_refused(handmade("[1, 2]"), "not a my_torch network");
    assert_refused(handmade("{\"format\": \"my_torch\", "), "invalid JSON");
}

Test(network_file, refuses_a_truncated_or_damaged_file)
{
    std::string path = temporary_path("damaged.nn");
    my_torch::Network network = small_network();

    my_torch::save_network(path, network);
    std::string bytes = read_bytes(path);
    write_bytes(path, bytes.substr(0, bytes.size() / 2));
    assert_refused(path, "is not a valid my_torch network");
    bytes[bytes.size() / 2] ^= 0x01;
    write_bytes(path, bytes);
    assert_refused(path, "wrong CRC-32");
}

Test(network_file, refuses_an_unknown_version)
{
    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 2, " + ONE_LAYER + "}", one_layer_arrays()), "version 2 is not supported");
    assert_refused(handmade("{\"format\": \"my_torch\", " + ONE_LAYER + "}", one_layer_arrays()), "header.version is missing");
}

Test(network_file, loads_a_handmade_network)
{
    my_torch::SavedNetwork loaded = my_torch::load_network(handmade("{\"format\": \"my_torch\", \"version\": 1, " + ONE_LAYER + "}", one_layer_arrays()));

    cr_assert_eq(loaded.network(Matrix::row({1, 1}))(0, 0), 6);
}

Test(network_file, refuses_unknown_layers_and_activations)
{
    std::string layers = "\"layers\": [{\"type\": \"conv\", \"activation\": \"relu\", \"shape\": [2, 1]}]";
    std::string activation = "\"layers\": [{\"type\": \"dense\", \"activation\": \"gelu\", \"shape\": [2, 1]}]";

    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 1, " + layers + "}", one_layer_arrays()), "unknown layer type 'conv'");
    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 1, " + activation + "}", one_layer_arrays()), "unknown activation 'gelu'");
    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 1, \"layers\": []}"), "at least one layer");
}

Test(network_file, refuses_arrays_that_do_not_match_the_header)
{
    std::string header = "{\"format\": \"my_torch\", \"version\": 1, " + ONE_LAYER + "}";
    my_torch::ZipEntries wrong_shape = {{"layer_0_weights.npy", my_torch::to_npy(Matrix({{1, 2}}))}, one_layer_arrays()[1]};
    my_torch::ZipEntries no_biases = {one_layer_arrays()[0]};
    my_torch::ZipEntries not_npy = {{"layer_0_weights.npy", "hello"}, one_layer_arrays()[1]};

    assert_refused(handmade(header, wrong_shape), "layer_0_weights.npy has shape 1x2 but the header announces 2x1");
    assert_refused(handmade(header, no_biases), "layer_0_biases.npy is missing");
    assert_refused(handmade(header, not_npy), "layer_0_weights.npy: not a .npy array");
}

Test(network_file, refuses_layers_that_do_not_chain_or_bad_shapes)
{
    std::string two_layers = "\"layers\": [{\"type\": \"dense\", \"activation\": \"relu\", \"shape\": [2, 1]}, {\"type\": \"dense\", \"activation\": \"relu\", \"shape\": [3, 1]}]";
    my_torch::ZipEntries arrays = one_layer_arrays();

    arrays.emplace_back("layer_1_weights.npy", my_torch::to_npy(Matrix(3, 1)));
    arrays.emplace_back("layer_1_biases.npy", my_torch::to_npy(Matrix::row({0}), true));
    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 1, " + two_layers + "}", arrays), "layer 1 expects 3 inputs but receives 1");
    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 1, \"layers\": [{\"type\": \"dense\", \"activation\": \"relu\", \"shape\": [2.5, 1]}]}"), "must be a whole number");
    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 1, \"layers\": [{\"type\": \"dense\", \"activation\": \"relu\", \"shape\": [0, 1]}]}"), "at least 1x1");
    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 1, \"layers\": [{\"type\": \"dense\", \"activation\": \"relu\", \"shape\": 2}]}"), "[n_inputs, n_outputs]");
    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 1, \"layers\": [{\"type\": \"dense\", \"shape\": [2, 1]}]}"), "layers[0].activation is missing");
    assert_refused(handmade("{\"format\": \"my_torch\", \"version\": 1, \"layers\": [3]}"), "layers[0] must be an object");
}

Test(network_file, refuses_bad_metadata_or_training)
{
    std::string start = "{\"format\": \"my_torch\", \"version\": 1, " + ONE_LAYER;
    std::string optimizer = "\"optimizer\": {\"name\": \"sgd\", \"learning_rate\": 0.1, \"schedule\": {\"name\": \"constant\"}}";
    std::string loss = "\"loss\": {\"name\": \"cross_entropy\"}";
    std::string rest = ", \"epoch\": 1, \"losses\": [0.5], \"batch_size\": 4, \"random_state\": ";

    assert_refused(handmade(start + ", \"metadata\": [1]}", one_layer_arrays()), "header.metadata must be an object");
    assert_refused(handmade(start + ", \"training\": {" + loss + ", " + optimizer + rest + "\"1 2 3\"}}", one_layer_arrays()), "random_state");
    assert_refused(handmade(start + ", \"training\": {\"loss\": {\"name\": \"hinge\"}, " + optimizer + rest + "\"\"}}", one_layer_arrays()), "unknown loss 'hinge'");
    assert_refused(handmade(start + ", \"training\": {" + loss + ", \"optimizer\": {\"name\": \"adam\", \"learning_rate\": 0.1, \"schedule\": {\"name\": \"constant\"}}" + rest + "\"\"}}", one_layer_arrays()), "unknown optimizer 'adam'");
    assert_refused(handmade(start + ", \"training\": {" + loss + ", \"optimizer\": {\"name\": \"sgd\", \"learning_rate\": 0.1, \"schedule\": {\"name\": \"cosine\"}}" + rest + "\"\"}}", one_layer_arrays()), "unknown schedule 'cosine'");
    assert_refused(handmade(start + ", \"training\": {" + loss + ", \"optimizer\": {\"name\": \"sgd\", \"learning_rate\": -1, \"schedule\": {\"name\": \"constant\"}}" + rest + "\"\"}}", one_layer_arrays()), "learning rate must be positive");
    assert_refused(handmade(start + ", \"training\": {" + loss + ", " + optimizer + ", \"epoch\": 1, \"losses\": [\"a\"], \"batch_size\": 4, \"random_state\": \"\"}}", one_layer_arrays()), "losses must be a list of numbers");
}

Test(network_file, a_momentum_training_needs_its_velocities)
{
    std::string path = temporary_path("velocities.nn");
    my_torch::Network network = small_network();
    my_torch::Training training = momentum_training();

    training.run(network, INPUTS, LABELS, 1);
    my_torch::save_network(path, network, Json::Object(), &training);
    my_torch::ZipEntries entries = my_torch::from_zip(read_bytes(path));
    entries.pop_back();
    write_bytes(path, my_torch::to_zip(entries));
    assert_refused(path, "optimizer_5.npy is missing");
}

/* Overfitting guards */

static my_torch::Network network_with_dropout()
{
    std::mt19937 rng(1);
    std::vector<std::unique_ptr<my_torch::Layer>> layers;
    layers.push_back(my_torch::DenseLayer::create(3, 5, std::make_unique<my_torch::ReLU>(), rng));
    layers.push_back(std::make_unique<my_torch::Dropout>(0.3, std::mt19937(9)));
    layers.push_back(my_torch::DenseLayer::create(5, 2, std::make_unique<my_torch::Softmax>(), rng));
    return my_torch::Network(std::move(layers));
}

Test(network_file, keeps_dropout_layers_with_their_rate_and_random_state)
{
    std::string path = temporary_path("dropout.nn");
    my_torch::Network network = network_with_dropout();
    my_torch::Training training = momentum_training();

    training.run(network, INPUTS, LABELS, 2);
    my_torch::save_network(path, network, Json::Object(), &training);
    my_torch::SavedNetwork loaded = my_torch::load_network(path);
    my_torch::Dropout &dropout = dynamic_cast<my_torch::Dropout &>(*loaded.network.layers[1]);
    cr_assert_eq(dropout.rate, 0.3);
    cr_assert(dropout.rng == dynamic_cast<my_torch::Dropout &>(*network.layers[1]).rng);
    assert_same_bits(dense(loaded.network, 2).weights, dense(network, 2).weights);
    assert_same_bits(loaded.network(INPUTS), network(INPUTS));
    my_torch::ZipEntries entries = my_torch::from_zip(read_bytes(path));
    cr_assert_str_eq(entries[3].first.c_str(), "layer_2_weights.npy");
}

Test(network_file, a_resumed_training_with_dropout_gives_the_same_network)
{
    std::string path = temporary_path("resume_dropout.nn");
    my_torch::Network in_one_go = network_with_dropout();
    my_torch::Network first_half = network_with_dropout();
    my_torch::Training one_go = momentum_training();
    my_torch::Training first = momentum_training();

    one_go.run(in_one_go, INPUTS, LABELS, 8);
    first.run(first_half, INPUTS, LABELS, 4);
    my_torch::save_network(path, first_half, Json::Object(), &first);
    my_torch::SavedNetwork resumed = my_torch::load_network(path);
    resumed.training->run(resumed.network, INPUTS, LABELS, 4);
    assert_same_bits(dense(resumed.network, 0).weights, dense(in_one_go, 0).weights);
    assert_same_bits(dense(resumed.network, 2).weights, dense(in_one_go, 2).weights);
}

Test(network_file, keeps_the_weight_decay_and_the_early_stopping_settings)
{
    std::string path = temporary_path("guards.nn");
    my_torch::Network network = small_network();
    my_torch::Training training = momentum_training();

    training.optimizer->weight_decay = 1e-4;
    training.validation_split = 0.2;
    training.patience = 7;
    training.max_epochs = 3;
    training.fit(network, INPUTS, LABELS);
    my_torch::save_network(path, network, Json::Object(), &training);
    my_torch::SavedNetwork loaded = my_torch::load_network(path);
    my_torch::Training &read = *loaded.training;
    cr_assert_eq(read.optimizer->weight_decay, 1e-4);
    cr_assert_eq(read.validation_split, 0.2);
    cr_assert_eq(read.patience, 7);
    cr_assert_eq(read.max_epochs, 3);
    cr_assert(read.validation_losses == training.validation_losses);
    cr_assert_eq(read.validation_losses.size(), 3);
}

Test(network_file, an_older_file_without_the_new_settings_takes_the_defaults)
{
    std::string header = "{\"format\": \"my_torch\", \"version\": 1, " + ONE_LAYER + ", \"training\": {\"epoch\": 0, \"losses\": [], \"batch_size\": 4, "
        "\"loss\": {\"name\": \"cross_entropy\"}, \"optimizer\": {\"name\": \"sgd\", \"learning_rate\": 0.1, \"schedule\": {\"name\": \"constant\"}}, "
        "\"random_state\": \"" + [] { std::ostringstream state; state << std::mt19937(); return state.str(); }() + "\"}}";
    my_torch::SavedNetwork loaded = my_torch::load_network(handmade(header, one_layer_arrays()));

    cr_assert_eq(loaded.training->optimizer->weight_decay, 0.0);
    cr_assert_eq(loaded.training->validation_split, 0.1);
    cr_assert_eq(loaded.training->patience, 5);
    cr_assert_eq(loaded.training->max_epochs, 50);
    cr_assert(loaded.training->validation_losses.empty());
}

Test(network_file, refuses_a_bad_dropout_layer)
{
    std::string start = "{\"format\": \"my_torch\", \"version\": 1, \"layers\": [{\"type\": \"dense\", \"activation\": \"relu\", \"shape\": [2, 1]}, ";

    assert_refused(handmade(start + "{\"type\": \"dropout\", \"rate\": 1.5}]}", one_layer_arrays()), "dropout rate must be in [0, 1)");
    assert_refused(handmade(start + "{\"type\": \"dropout\"}]}", one_layer_arrays()), "layers[1].rate is missing");
    assert_refused(handmade(start + "{\"type\": \"dropout\", \"rate\": 0.1, \"random_state\": \"x\"}]}", one_layer_arrays()), "layers[1].random_state");
    cr_assert_eq(dynamic_cast<my_torch::Dropout &>(*my_torch::load_network(handmade(start + "{\"type\": \"dropout\", \"rate\": 0.1}]}", one_layer_arrays())).network.layers[1]).rate, 0.1);
}

Test(network_file, the_loss_and_optimizer_readers_can_be_used_on_their_own)
{
    my_torch::Json loss = my_torch::Json::parse("{\"name\": \"cross_entropy\", \"epsilon\": 0.001}");
    my_torch::Json optimizer = my_torch::Json::parse("{\"name\": \"momentum\", \"learning_rate\": 0.02, \"momentum\": 0.5, \"weight_decay\": 0.001, \"schedule\": {\"name\": \"step_decay\", \"factor\": 0.5, \"every\": 4}}");
    std::unique_ptr<my_torch::Optimizer> built = my_torch::build_optimizer(optimizer);

    cr_assert(my_torch::build_loss(loss)->name == "cross_entropy");
    cr_assert(built->name == "momentum");
    cr_assert_float_eq(built->initial_learning_rate, 0.02, 1e-15);
    cr_assert_float_eq(built->weight_decay, 0.001, 1e-15);
    cr_assert(built->schedule->name == "step_decay");
    cr_assert_throw(my_torch::build_loss(my_torch::Json::parse("{\"name\": \"hinge\"}")), std::runtime_error);
}
