/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A config's layers are dense (outputs, activation, initialization) or dropout (rate); its training
** reuses the .nn reader of losses and optimizers, and every setting it leaves out keeps the engine's;
** the generator checks every config before writing any file
*/

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "engine/engine.hpp"
#include "generator/generator.hpp"
#include "io/network_file/network_file.hpp"
#include "model/activations/activation.hpp"
#include "model/initialization/initialization.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "model/layers/layers/dropout/dropout.hpp"

namespace my_torch {

static const int EXIT_SUCCESS_CODE = 0;
static const int EXIT_ERROR_CODE = 84;

static const char *USAGE = "USAGE\n"
    "    ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]\n"
    "\n"
    "DESCRIPTION\n"
    "    config_file_i    Configuration file containing description of a neural network we want to generate.\n"
    "    nb_i             Number of neural networks to generate based on the configuration file.\n";

static const Json &member(const Json &object, const std::string &key, const std::string &where)
{
    if (!object.is_object())
        throw ConfigError(where + " must be an object");
    const Json *value = object.find(key);
    if (!value)
        throw ConfigError(where + "." + key + " is missing");
    return *value;
}

static double number(const Json &object, const std::string &key, const std::string &where)
{
    const Json &value = member(object, key, where);
    if (!value.is_number())
        throw ConfigError(where + "." + key + " must be a number");
    return value.as_number();
}

static std::size_t whole(const Json &object, const std::string &key, const std::string &where)
{
    double value = number(object, key, where);
    if (!(value >= 1 && value <= 9007199254740992.0) || std::floor(value) != value)
        throw ConfigError(where + "." + key + " must be a whole number of at least 1, got " + Json(value).dump());
    return static_cast<std::size_t>(value);
}

static std::string text_or(const Json &object, const std::string &key, const std::string &where, const std::string &fallback)
{
    const Json *value = object.find(key);
    if (!value)
        return fallback;
    if (!value->is_string())
        throw ConfigError(where + "." + key + " must be a string");
    return value->as_string();
}

static std::unique_ptr<Layer> dense_layer(const Json &description, std::size_t &width, std::mt19937 &rng, const std::string &where)
{
    std::size_t outputs = whole(description, "outputs", where);
    std::unique_ptr<Activation> activation = Activation::from_name(text_or(description, "activation", where, "linear"));
    Initialization initialization = initialization_from_name(text_or(description, "initialization", where, "auto"));
    std::unique_ptr<Layer> layer = DenseLayer::create(width, outputs, std::move(activation), rng, initialization);
    width = outputs;
    return layer;
}

static std::unique_ptr<Layer> dropout_layer(const Json &description, std::mt19937 &rng, const std::string &where)
{
    double rate = number(description, "rate", where);
    if (!(rate >= 0.0 && rate < 1.0))
        throw ConfigError(where + ".rate must be in [0, 1), got " + Json(rate).dump());
    return std::make_unique<Dropout>(rate, std::mt19937(rng()));
}

static std::unique_ptr<Layer> config_layer(const Json &description, std::size_t &width, std::mt19937 &rng, const std::string &where)
{
    const std::string type = text_or(description, "type", where, "dense");
    if (type == "dense")
        return dense_layer(description, width, rng, where);
    if (type == "dropout")
        return dropout_layer(description, rng, where);
    throw ConfigError(where + ": unknown layer type '" + type + "', expected dense or dropout");
}

static Network config_network(const Json &config, std::mt19937 &rng)
{
    std::size_t width = whole(config, "inputs", "config");
    const Json descriptions = member(config, "layers", "config");
    if (!descriptions.is_array() || descriptions.as_array().empty())
        throw ConfigError("config.layers must be a list of at least one layer");
    std::vector<std::unique_ptr<Layer>> layers;
    for (std::size_t position = 0; position < descriptions.as_array().size(); position++)
        layers.push_back(config_layer(descriptions.as_array()[position], width, rng, "config.layers[" + std::to_string(position) + "]"));
    return Network(std::move(layers));
}

static Json with_schedule(const Json &optimizer)
{
    Json description = optimizer;
    if (description.is_object() && !description.find("schedule"))
        description.set("schedule", Json(Json::Object {{"name", "constant"}}));
    return description;
}

static void apply_training_settings(Training &training, const Json &settings)
{
    const std::string where = "config.training";
    if (settings.find("batch_size"))
        training.batch_size = whole(settings, "batch_size", where);
    if (settings.find("loss"))
        training.loss = build_loss(*settings.find("loss"));
    if (settings.find("optimizer"))
        training.optimizer = build_optimizer(with_schedule(*settings.find("optimizer")));
    if (settings.find("validation_split"))
        training.validation_split = number(settings, "validation_split", where);
    if (settings.find("patience"))
        training.patience = whole(settings, "patience", where);
    if (settings.find("max_epochs"))
        training.max_epochs = whole(settings, "max_epochs", where);
}

static Training config_training(const Json &config, unsigned int seed)
{
    Training training = default_training();
    training.rng = std::mt19937(seed);
    if (const Json *settings = config.find("training")) {
        if (!settings->is_object())
            throw ConfigError("config.training must be an object");
        apply_training_settings(training, *settings);
    }
    return training;
}

static Json config_metadata(const Json &config, unsigned int seed)
{
    const Json *given = config.find("metadata");
    if (given && !given->is_object())
        throw ConfigError("config.metadata must be an object");
    Json metadata = given ? *given : Json(Json::Object());
    metadata.set("seed", Json(static_cast<std::size_t>(seed)));
    return metadata;
}

static unsigned int config_seed(const Json &config)
{
    const Json *seed = config.find("seed");
    if (!seed)
        return 0;
    if (!seed->is_number() || seed->as_number() < 0 || std::floor(seed->as_number()) != seed->as_number())
        throw ConfigError("config.seed must be a whole number of at least 0");
    return static_cast<unsigned int>(seed->as_number());
}

Json read_config(const std::string &path)
{
    std::ifstream file(path);
    if (!file)
        throw ConfigError("cannot open the config file '" + path + "'");
    std::stringstream text;
    text << file.rdbuf();
    try {
        return Json::parse(text.str());
    } catch (const std::exception &error) {
        throw ConfigError("'" + path + "' is not valid JSON: " + error.what());
    }
}

GeneratedNetwork generate_network(const Json &config, unsigned int seed)
{
    if (!config.is_object())
        throw ConfigError("a config must be a JSON object");
    std::mt19937 rng(seed);
    try {
        Network network = config_network(config, rng);
        return {std::move(network), config_training(config, seed), config_metadata(config, seed)};
    } catch (const ConfigError &) {
        throw;
    } catch (const std::exception &error) {
        throw ConfigError(error.what());
    }
}

std::string generated_name(const std::string &config_path, std::size_t number)
{
    return std::filesystem::path(config_path).stem().string() + "_" + std::to_string(number) + ".nn";
}

std::vector<std::string> generate_files(const std::string &config_path, std::size_t count, const std::string &out_dir)
{
    Json config = read_config(config_path);
    unsigned int first_seed = config_seed(config);
    std::vector<std::string> paths;
    for (std::size_t number = 1; number <= count; number++) {
        unsigned int seed = first_seed + static_cast<unsigned int>(number - 1);
        GeneratedNetwork generated = generate_network(config, seed);
        generated.metadata.set("config", Json(std::filesystem::path(config_path).filename().string()));
        std::string path = (std::filesystem::path(out_dir) / generated_name(config_path, number)).string();
        save_network(path, generated.network, generated.metadata, &generated.training);
        paths.push_back(path);
    }
    return paths;
}

static std::size_t network_count(const std::string &text)
{
    std::size_t used = 0;
    unsigned long long count = 0;
    try {
        count = std::stoull(text, &used);
    } catch (const std::exception &) {
        used = 0;
    }
    if (used != text.size() || text.empty() || text[0] == '-' || count < 1)
        throw ConfigError("the number of networks must be a whole number of at least 1, got '" + text + "'");
    return static_cast<std::size_t>(count);
}

static void check_arguments(const std::vector<std::string> &arguments)
{
    if (arguments.empty() || arguments.size() % 2 != 0)
        throw ConfigError(std::string("expected pairs of config file and number of networks\n") + USAGE);
    for (std::size_t index = 0; index < arguments.size(); index += 2) {
        network_count(arguments[index + 1]);
        Json config = read_config(arguments[index]);
        generate_network(config, config_seed(config));
    }
}

int run_generator(const std::vector<std::string> &arguments, std::ostream &out, std::ostream &err)
{
    if (arguments.size() == 1 && (arguments[0] == "--help" || arguments[0] == "-h")) {
        out << USAGE;
        return EXIT_SUCCESS_CODE;
    }
    try {
        check_arguments(arguments);
        for (std::size_t index = 0; index < arguments.size(); index += 2)
            generate_files(arguments[index], network_count(arguments[index + 1]));
    } catch (const std::exception &error) {
        err << "my_torch_generator: " << error.what() << std::endl;
        return EXIT_ERROR_CODE;
    }
    return EXIT_SUCCESS_CODE;
}

}
