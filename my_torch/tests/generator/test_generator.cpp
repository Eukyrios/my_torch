/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of generating blank networks from a configuration, in code and as .nn files
*/

#include <cmath>
#include <criterion/criterion.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

#include "engine/engine.hpp"
#include "generator/generator.hpp"
#include "io/network_file/network_file.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "model/layers/layers/dropout/dropout.hpp"

static const char *CONFIG = R"({
    "inputs": 769,
    "seed": 7,
    "layers": [
        {"type": "dense", "outputs": 47, "activation": "relu", "initialization": "he"},
        {"type": "dropout", "rate": 0.2},
        {"type": "dense", "outputs": 5, "activation": "softmax", "initialization": "lecun"}
    ],
    "training": {"batch_size": 16, "optimizer": {"name": "momentum", "learning_rate": 0.02, "momentum": 0.5}},
    "metadata": {"notes": "test network"}
})";

static my_torch::Json config()
{
    return my_torch::Json::parse(CONFIG);
}

static my_torch::Json config_with(const std::string &key, const std::string &value)
{
    my_torch::Json changed = config();
    changed.set(key, my_torch::Json::parse(value));
    return changed;
}

static const my_torch::DenseLayer &dense(const my_torch::Network &network, std::size_t position)
{
    return dynamic_cast<const my_torch::DenseLayer &>(*network.layers[position]);
}

static double standard_deviation(const my_torch::Matrix &values)
{
    double mean = values.sum() / values.size();
    double total = 0.0;
    for (std::size_t row = 0; row < values.rows(); row++)
        for (std::size_t col = 0; col < values.cols(); col++)
            total += (values(row, col) - mean) * (values(row, col) - mean);
    return std::sqrt(total / values.size());
}

static std::string temporary_folder()
{
    std::string folder = "/tmp/my_torch_generator_test_" + std::to_string(getpid());
    std::filesystem::create_directories(folder);
    return folder;
}

static std::string write_config(const std::string &folder, const std::string &name, const std::string &text)
{
    std::string path = folder + "/" + name;
    std::ofstream(path) << text;
    return path;
}

Test(generator, builds_the_layers_of_the_config)
{
    my_torch::GeneratedNetwork generated = my_torch::generate_network(config(), 0);

    cr_assert_eq(generated.network.layers.size(), 3u);
    cr_assert(generated.network.shape() == my_torch::Shape(769, 5));
    cr_assert(dense(generated.network, 0).activation->name == "relu");
    cr_assert(dense(generated.network, 2).activation->name == "softmax");
    cr_assert_float_eq(dynamic_cast<const my_torch::Dropout &>(*generated.network.layers[1]).rate, 0.2, 1e-15);
}

Test(generator, uses_the_initialization_of_each_layer)
{
    my_torch::GeneratedNetwork generated = my_torch::generate_network(config(), 0);

    cr_assert_float_eq(standard_deviation(dense(generated.network, 0).weights), std::sqrt(2.0 / 769), 0.003);
    cr_assert_float_eq(standard_deviation(dense(generated.network, 2).weights), std::sqrt(1.0 / 47), 0.04);
}

Test(generator, the_same_seed_gives_the_same_network_and_another_seed_a_different_one)
{
    my_torch::Matrix first = dense(my_torch::generate_network(config(), 3).network, 0).weights;
    my_torch::Matrix again = dense(my_torch::generate_network(config(), 3).network, 0).weights;
    my_torch::Matrix other = dense(my_torch::generate_network(config(), 4).network, 0).weights;

    cr_assert(first(0, 0) == again(0, 0) && first(768, 46) == again(768, 46));
    cr_assert(first(0, 0) != other(0, 0));
}

Test(generator, applies_the_training_settings_and_keeps_the_defaults_for_the_rest)
{
    my_torch::GeneratedNetwork generated = my_torch::generate_network(config(), 0);
    my_torch::Training defaults = my_torch::default_training();

    cr_assert_eq(generated.training.batch_size, 16u);
    cr_assert_float_eq(generated.training.optimizer->initial_learning_rate, 0.02, 1e-15);
    cr_assert(generated.training.optimizer->schedule->name == "constant");
    cr_assert(generated.training.loss->name == defaults.loss->name);
    cr_assert_float_eq(generated.training.validation_split, defaults.validation_split, 1e-15);
    cr_assert_eq(generated.training.max_epochs, defaults.max_epochs);
    cr_assert_eq(generated.training.epoch, 0u);
}

Test(generator, without_a_training_section_uses_the_engine_training)
{
    my_torch::Json bare = config();
    bare.set("training", my_torch::Json(my_torch::Json::Object()));
    my_torch::GeneratedNetwork generated = my_torch::generate_network(bare, 0);
    my_torch::Training defaults = my_torch::default_training();

    cr_assert_eq(generated.training.batch_size, defaults.batch_size);
    cr_assert_float_eq(generated.training.optimizer->initial_learning_rate, defaults.optimizer->initial_learning_rate, 1e-15);
}

Test(generator, keeps_the_metadata_and_adds_the_seed)
{
    my_torch::GeneratedNetwork generated = my_torch::generate_network(config(), 9);

    cr_assert(generated.metadata.find("notes")->as_string() == "test network");
    cr_assert_float_eq(generated.metadata.find("seed")->as_number(), 9.0, 1e-15);
}

Test(generator, names_the_files_after_the_config)
{
    cr_assert(my_torch::generated_name("configs/basic_network.conf", 2) == "basic_network_2.nn");
    cr_assert(my_torch::generated_name("medium", 1) == "medium_1.nn");
}

Test(generator, writes_networks_that_load_back_with_their_training)
{
    std::string folder = temporary_folder();
    std::string config_path = write_config(folder, "basic_network.conf", CONFIG);
    std::vector<std::string> paths = my_torch::generate_files(config_path, 3, folder);

    cr_assert_eq(paths.size(), 3u);
    cr_assert(paths[2] == folder + "/basic_network_3.nn");
    my_torch::SavedNetwork first = my_torch::load_network(paths[0]);
    my_torch::SavedNetwork second = my_torch::load_network(paths[1]);
    cr_assert(first.network.shape() == my_torch::Shape(769, 5));
    cr_assert(first.training.has_value() && first.training->batch_size == 16u);
    cr_assert_float_eq(first.metadata.find("seed")->as_number(), 7.0, 1e-15);
    cr_assert_float_eq(second.metadata.find("seed")->as_number(), 8.0, 1e-15);
    cr_assert(first.metadata.find("config")->as_string() == "basic_network.conf");
    cr_assert(dense(first.network, 0).weights(0, 0) != dense(second.network, 0).weights(0, 0));
    std::filesystem::remove_all(folder);
}

Test(generator, rejects_configs_it_cannot_build)
{
    const char *broken[][2] = {
        {"inputs", "0"},
        {"inputs", "\"many\""},
        {"layers", "[]"},
        {"layers", "[{\"type\": \"dense\", \"outputs\": 0}]"},
        {"layers", "[{\"type\": \"conv\", \"outputs\": 5, \"activation\": \"relu\"}]"},
        {"layers", "[{\"type\": \"dense\", \"outputs\": 5, \"activation\": \"swish\"}]"},
        {"layers", "[{\"type\": \"dense\", \"outputs\": 5, \"activation\": \"relu\", \"initialization\": \"orthogonal\"}]"},
        {"layers", "[{\"type\": \"dense\", \"outputs\": 5}]"},
        {"layers", "[{\"type\": \"dropout\", \"rate\": 1.5}, {\"type\": \"dense\", \"outputs\": 5, \"activation\": \"softmax\"}]"},
        {"training", "{\"optimizer\": {\"name\": \"adam\", \"learning_rate\": 0.1}}"},
        {"training", "{\"batch_size\": 0}"},
        {"metadata", "[1, 2]"},
    };
    for (const auto &[key, value] : broken)
        cr_assert_throw(my_torch::generate_network(config_with(key, value), 0), my_torch::ConfigError, "%s = %s", key, value);
    cr_assert_throw(my_torch::generate_network(my_torch::Json::parse("[]"), 0), my_torch::ConfigError);
}

Test(generator, read_config_rejects_missing_or_invalid_files)
{
    std::string folder = temporary_folder();

    cr_assert_throw(my_torch::read_config(folder + "/missing.conf"), my_torch::ConfigError);
    cr_assert_throw(my_torch::read_config(write_config(folder, "broken.conf", "{\"inputs\": ")), my_torch::ConfigError);
    std::filesystem::remove_all(folder);
}

static int run(const std::vector<std::string> &arguments, std::string &out, std::string &err)
{
    std::ostringstream out_stream;
    std::ostringstream err_stream;
    int status = my_torch::run_generator(arguments, out_stream, err_stream);
    out = out_stream.str();
    err = err_stream.str();
    return status;
}

Test(generator, help_prints_the_usage_and_exits_0)
{
    std::string out;
    std::string err;

    cr_assert_eq(run({"--help"}, out, err), 0);
    cr_assert(out.rfind("USAGE\n    ./my_torch_generator config_file_1 nb_1", 0) == 0);
    cr_assert(err.empty());
}

Test(generator, writes_nb_networks_per_config_in_the_current_folder)
{
    std::string folder = temporary_folder();
    std::string basic = write_config(folder, "basic_network.conf", CONFIG);
    std::string other = write_config(folder, "other.conf", CONFIG);
    std::filesystem::path previous = std::filesystem::current_path();
    std::string out;
    std::string err;

    std::filesystem::current_path(folder);
    int status = run({basic, "3", other, "1"}, out, err);
    std::filesystem::current_path(previous);
    cr_assert_eq(status, 0, "%s", err.c_str());
    for (const char *name : {"basic_network_1.nn", "basic_network_2.nn", "basic_network_3.nn", "other_1.nn"})
        cr_assert(std::filesystem::exists(folder + "/" + name), "%s", name);
    cr_assert_not(std::filesystem::exists(folder + "/other_2.nn"));
    std::filesystem::remove_all(folder);
}

Test(generator, bad_arguments_print_an_error_and_exit_84_without_writing_anything)
{
    std::string folder = temporary_folder();
    std::string basic = write_config(folder, "basic_network.conf", CONFIG);
    std::string out;
    std::string err;

    for (const std::vector<std::string> &arguments : std::vector<std::vector<std::string>>{{}, {basic}, {basic, "0"}, {basic, "two"}, {basic, "-1"}, {basic, "2", folder + "/missing.conf", "1"}}) {
        cr_assert_eq(run(arguments, out, err), 84);
        cr_assert(err.rfind("my_torch_generator: ", 0) == 0, "%s", err.c_str());
    }
    cr_assert_not(std::filesystem::exists("basic_network_1.nn"));
    std::filesystem::remove_all(folder);
}
