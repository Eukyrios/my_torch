/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A .nn file is an uncompressed zip holding header.json (format, version, layers, metadata and
** the training) and one .npy per array: layer_<i>_weights and layer_<i>_biases of each dense
** layer i (a dropout layer has no array, only its rate and random state in the header), and
** optimizer_<i> for the optimizer's memory of parameter i (Momentum's velocities). Settings added
** after the first networks (dropout, weight decay, early stopping) take their default value when
** a file does not have them, so older files still load. Saving writes a temporary
** file then renames it, so a failed save never breaks the network already there; loading checks
** every part before building anything
*/

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>

#include "io/network_file/network_file.hpp"
#include "io/npy/npy.hpp"
#include "io/zip/zip.hpp"
#include "model/activations/activation.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "model/layers/layers/dropout/dropout.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/losses/losses/mean_squared_error/mean_squared_error.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/optimizers/optimizers/sgd/sgd.hpp"
#include "training/schedules/schedules/constant/constant.hpp"
#include "training/schedules/schedules/exponential_decay/exponential_decay.hpp"
#include "training/schedules/schedules/step_decay/step_decay.hpp"
#include "training/schedules/schedules/time_decay/time_decay.hpp"

namespace my_torch {

static const std::string HEADER = "header.json";

static std::string weights_name(std::size_t layer) { return "layer_" + std::to_string(layer) + "_weights.npy"; }
static std::string biases_name(std::size_t layer) { return "layer_" + std::to_string(layer) + "_biases.npy"; }
static std::string optimizer_name(std::size_t parameter) { return "optimizer_" + std::to_string(parameter) + ".npy"; }

static std::string random_state(const std::mt19937 &rng)
{
    std::ostringstream state;
    state << rng;
    return state.str();
}

/* Saving: everything is described in the header */

static Json describe_layer(const Layer &layer, std::size_t position)
{
    if (const DenseLayer *dense = dynamic_cast<const DenseLayer *>(&layer))
        return Json::Object {
            {"type", dense->name},
            {"activation", dense->activation->name},
            {"shape", Json::Array {dense->weights.rows(), dense->weights.cols()}},
        };
    if (const Dropout *dropout = dynamic_cast<const Dropout *>(&layer))
        return Json::Object {{"type", dropout->name}, {"rate", dropout->rate}, {"random_state", random_state(dropout->rng)}};
    throw NetworkFileError("cannot save layer " + std::to_string(position) + " of type '" + layer.name + "', only dense and dropout layers can be saved");
}

static Json describe_layers(const Network &network)
{
    Json::Array layers;
    for (std::size_t position = 0; position < network.layers.size(); position++)
        layers.push_back(describe_layer(*network.layers[position], position));
    return layers;
}

static Json describe_loss(const Loss &loss)
{
    Json description = Json::Object {{"name", loss.name}};
    if (const CrossEntropy *cross_entropy = dynamic_cast<const CrossEntropy *>(&loss))
        description.set("epsilon", cross_entropy->epsilon);
    return description;
}

static Json describe_schedule(const Schedule &schedule)
{
    Json description = Json::Object {{"name", schedule.name}};
    if (const TimeDecay *time_decay = dynamic_cast<const TimeDecay *>(&schedule))
        description.set("decay", time_decay->decay);
    if (const ExponentialDecay *exponential = dynamic_cast<const ExponentialDecay *>(&schedule))
        description.set("factor", exponential->factor);
    if (const StepDecay *step_decay = dynamic_cast<const StepDecay *>(&schedule)) {
        description.set("factor", step_decay->factor);
        description.set("every", step_decay->every);
    }
    return description;
}

static Json describe_optimizer(const Optimizer &optimizer)
{
    Json description = Json::Object {
        {"name", optimizer.name},
        {"learning_rate", optimizer.initial_learning_rate},
        {"current_learning_rate", optimizer.learning_rate},
    };
    if (const Momentum *momentum = dynamic_cast<const Momentum *>(&optimizer))
        description.set("momentum", momentum->momentum);
    description.set("weight_decay", optimizer.weight_decay);
    description.set("schedule", describe_schedule(*optimizer.schedule));
    return description;
}

static Json describe_training(const Training &training)
{
    Json::Array losses(training.losses.begin(), training.losses.end());
    Json::Array validation_losses(training.validation_losses.begin(), training.validation_losses.end());
    return Json::Object {
        {"epoch", training.epoch},
        {"losses", losses},
        {"validation_losses", validation_losses},
        {"batch_size", training.batch_size},
        {"validation_split", training.validation_split},
        {"patience", training.patience},
        {"max_epochs", training.max_epochs},
        {"loss", describe_loss(*training.loss)},
        {"optimizer", describe_optimizer(*training.optimizer)},
        {"random_state", random_state(training.rng)},
    };
}

static void add_optimizer_memory(ZipEntries &entries, Network &network, const Optimizer &optimizer)
{
    const Momentum *momentum = dynamic_cast<const Momentum *>(&optimizer);
    if (!momentum)
        return;
    std::vector<Parameter> parameters = network.parameters();
    for (std::size_t index = 0; index < parameters.size(); index++) {
        auto found = momentum->velocities.find(parameters[index].value);
        // a parameter not updated yet has a zero velocity
        Matrix velocity = found == momentum->velocities.end() ? Matrix(parameters[index].value->rows(), parameters[index].value->cols()) : found->second;
        entries.emplace_back(optimizer_name(index), to_npy(velocity));
    }
}

static void write_file(const std::string &path, const std::string &bytes)
{
    std::string temporary = path + ".tmp";
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file || !file.write(bytes.data(), bytes.size()))
            throw NetworkFileError("cannot write the network to '" + path + "'");
    }
    std::error_code error;
    std::filesystem::rename(temporary, path, error);
    if (error) {
        std::filesystem::remove(temporary, error);
        throw NetworkFileError("cannot write the network to '" + path + "'");
    }
}

void save_network(const std::string &path, Network &network, const Json &metadata, const Training *training)
{
    if (!metadata.is_object())
        throw NetworkFileError("the metadata of a network must be a JSON object");
    Json header = Json::Object {
        {"format", NETWORK_FORMAT},
        {"version", NETWORK_FORMAT_VERSION},
        {"layers", describe_layers(network)},
        {"metadata", metadata},
    };
    if (training)
        header.set("training", describe_training(*training));
    ZipEntries entries = {{HEADER, header.dump() + "\n"}};
    for (std::size_t position = 0; position < network.layers.size(); position++)
        if (const DenseLayer *dense = dynamic_cast<const DenseLayer *>(network.layers[position].get())) {
            entries.emplace_back(weights_name(position), to_npy(dense->weights));
            entries.emplace_back(biases_name(position), to_npy(dense->biases, true));
        }
    if (training)
        add_optimizer_memory(entries, network, *training->optimizer);
    write_file(path, to_zip(entries));
}

/* Loading: every value is checked, errors name the part of the header at fault */

namespace {

class Invalid : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using Arrays = std::map<std::string, std::string>;

}

static Json member(const Json &object, const std::string &key, const std::string &where)
{
    if (!object.is_object())
        throw Invalid(where + " must be an object");
    const Json *value = object.find(key);
    if (!value)
        throw Invalid(where + "." + key + " is missing");
    return *value;
}

static double number(const Json &object, const std::string &key, const std::string &where)
{
    const Json value = member(object, key, where);
    if (!value.is_number())
        throw Invalid(where + "." + key + " must be a number");
    return value.as_number();
}

static std::size_t count(const Json &object, const std::string &key, const std::string &where)
{
    double value = number(object, key, where);
    if (!(value >= 0 && value <= 9007199254740992.0) || std::floor(value) != value)
        throw Invalid(where + "." + key + " must be a whole number, got " + Json(value).dump());
    return static_cast<std::size_t>(value);
}

static std::string text(const Json &object, const std::string &key, const std::string &where)
{
    const Json value = member(object, key, where);
    if (!value.is_string())
        throw Invalid(where + "." + key + " must be a string");
    return value.as_string();
}

static double number_or(const Json &object, const std::string &key, const std::string &where, double fallback)
{
    return object.find(key) ? number(object, key, where) : fallback;
}

static std::size_t count_or(const Json &object, const std::string &key, const std::string &where, std::size_t fallback)
{
    return object.find(key) ? count(object, key, where) : fallback;
}

static std::mt19937 build_random_state(const std::string &state, const std::string &where)
{
    std::istringstream text(state);
    std::mt19937 rng;
    text >> rng;
    if (text.fail() || !(text >> std::ws).eof())
        throw Invalid(where + ".random_state is not the state of a std::mt19937");
    return rng;
}

static std::string read_file(const std::string &path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file || std::filesystem::is_directory(path))
        throw NetworkFileError("cannot open the network file '" + path + "'");
    std::ostringstream bytes;
    bytes << file.rdbuf();
    return bytes.str();
}

static Arrays read_archive(const std::string &bytes)
{
    Arrays arrays;
    for (auto &[name, data] : from_zip(bytes))
        if (!arrays.emplace(name, data).second)
            throw Invalid("'" + name + "' is in the file twice");
    return arrays;
}

static Json read_header(const Arrays &arrays)
{
    auto found = arrays.find(HEADER);
    if (found == arrays.end())
        throw Invalid("no " + HEADER + ", not a my_torch network");
    Json header = Json::parse(found->second);
    if (!header.is_object() || !header.find("format") || *header.find("format") != Json(NETWORK_FORMAT))
        throw Invalid("not a my_torch network (format is not \"my_torch\")");
    const Json version = member(header, "version", "header");
    if (version != Json(NETWORK_FORMAT_VERSION))
        throw Invalid("version " + version.dump() + " is not supported, this my_torch reads version " + std::to_string(NETWORK_FORMAT_VERSION));
    return header;
}

static Matrix array(const Arrays &arrays, const std::string &name, std::size_t rows, std::size_t cols)
{
    auto found = arrays.find(name);
    if (found == arrays.end())
        throw Invalid(name + " is missing");
    Matrix matrix;
    try {
        matrix = from_npy(found->second);
    } catch (const NpyError &error) {
        throw Invalid(name + ": " + error.what());
    }
    if (matrix.rows() != rows || matrix.cols() != cols)
        throw Invalid(name + " has shape " + std::to_string(matrix.rows()) + "x" + std::to_string(matrix.cols()) + " but the header announces " + std::to_string(rows) + "x" + std::to_string(cols));
    return matrix;
}

static std::vector<double> row_values(const Matrix &row)
{
    std::vector<double> values;
    for (std::size_t col = 0; col < row.cols(); col++)
        values.push_back(row(0, col));
    return values;
}

static std::unique_ptr<Layer> build_dropout(const Json &description, const std::string &where)
{
    std::mt19937 rng = description.find("random_state") ? build_random_state(text(description, "random_state", where), where) : std::mt19937();
    return std::make_unique<Dropout>(number(description, "rate", where), rng);
}

static std::unique_ptr<Layer> build_layer(const Json &description, std::size_t position, const Arrays &arrays)
{
    std::string where = "layers[" + std::to_string(position) + "]";
    const std::string type = text(description, "type", where);
    if (type == "dropout")
        return build_dropout(description, where);
    if (type != "dense")
        throw Invalid(where + ": unknown layer type '" + type + "', expected dense or dropout");
    const Json shape = member(description, "shape", where);
    if (!shape.is_array() || shape.as_array().size() != 2)
        throw Invalid(where + ".shape must be [n_inputs, n_outputs]");
    Json sizes = Json::Object {{"n_inputs", shape.as_array()[0]}, {"n_outputs", shape.as_array()[1]}};
    std::size_t n_inputs = count(sizes, "n_inputs", where + ".shape");
    std::size_t n_outputs = count(sizes, "n_outputs", where + ".shape");
    if (n_inputs < 1 || n_outputs < 1)
        throw Invalid(where + ".shape must be at least 1x1");
    std::unique_ptr<Activation> activation = Activation::from_name(text(description, "activation", where));
    Matrix weights = array(arrays, weights_name(position), n_inputs, n_outputs);
    Matrix biases = array(arrays, biases_name(position), 1, n_outputs);
    return std::make_unique<DenseLayer>(weights, row_values(biases), std::move(activation));
}

static Network build_network(const Json &header, const Arrays &arrays)
{
    const Json &descriptions = member(header, "layers", "header");
    if (!descriptions.is_array() || descriptions.as_array().empty())
        throw Invalid("header.layers must be a list of at least one layer");
    std::vector<std::unique_ptr<Layer>> layers;
    for (std::size_t position = 0; position < descriptions.as_array().size(); position++)
        layers.push_back(build_layer(descriptions.as_array()[position], position, arrays));
    return Network(std::move(layers));
}

std::unique_ptr<Loss> build_loss(const Json &description)
{
    const std::string where = "training.loss";
    const std::string name = text(description, "name", where);
    if (name == "cross_entropy")
        return std::make_unique<CrossEntropy>(number_or(description, "epsilon", where, 1e-12));
    if (name == "mean_squared_error")
        return std::make_unique<MeanSquaredError>();
    throw Invalid(where + ": unknown loss '" + name + "', expected cross_entropy or mean_squared_error");
}

static std::unique_ptr<Schedule> build_schedule(const Json &description)
{
    const std::string where = "training.optimizer.schedule";
    const std::string name = text(description, "name", where);
    if (name == "constant")
        return std::make_unique<Constant>();
    if (name == "time_decay")
        return std::make_unique<TimeDecay>(number(description, "decay", where));
    if (name == "exponential_decay")
        return std::make_unique<ExponentialDecay>(number(description, "factor", where));
    if (name == "step_decay")
        return std::make_unique<StepDecay>(number(description, "factor", where), count(description, "every", where));
    throw Invalid(where + ": unknown schedule '" + name + "', expected constant, time_decay, exponential_decay or step_decay");
}

std::unique_ptr<Optimizer> build_optimizer(const Json &description)
{
    const std::string where = "training.optimizer";
    const std::string name = text(description, "name", where);
    double learning_rate = number(description, "learning_rate", where);
    std::unique_ptr<Schedule> schedule = build_schedule(member(description, "schedule", where));
    std::unique_ptr<Optimizer> optimizer;
    if (name == "sgd")
        optimizer = std::make_unique<SGD>(learning_rate, std::move(schedule));
    else if (name == "momentum")
        optimizer = std::make_unique<Momentum>(learning_rate, number(description, "momentum", where), std::move(schedule));
    else
        throw Invalid(where + ": unknown optimizer '" + name + "', expected sgd or momentum");
    optimizer->learning_rate = number_or(description, "current_learning_rate", where, learning_rate);
    optimizer->weight_decay = number_or(description, "weight_decay", where, 0.0);
    return optimizer;
}

static void restore_optimizer_memory(Optimizer &optimizer, Network &network, const Arrays &arrays)
{
    Momentum *momentum = dynamic_cast<Momentum *>(&optimizer);
    if (!momentum)
        return;
    std::vector<Parameter> parameters = network.parameters();
    for (std::size_t index = 0; index < parameters.size(); index++) {
        const Matrix &value = *parameters[index].value;
        momentum->velocities[&value] = array(arrays, optimizer_name(index), value.rows(), value.cols());
    }
}

static std::vector<double> build_losses(const Json &losses, const std::string &where)
{
    std::vector<double> values;
    if (!losses.is_array())
        throw Invalid(where + " must be a list of numbers");
    for (const Json &loss : losses.as_array()) {
        if (!loss.is_number())
            throw Invalid(where + " must be a list of numbers");
        values.push_back(loss.as_number());
    }
    return values;
}

static Training build_training(const Json &description, Network &network, const Arrays &arrays)
{
    const std::string where = "training";
    std::size_t epoch = count(description, "epoch", where);
    std::vector<double> losses = build_losses(member(description, "losses", where), where + ".losses");
    const Json *validation_losses = description.find("validation_losses");
    std::size_t batch_size = count(description, "batch_size", where);
    std::unique_ptr<Loss> loss = build_loss(member(description, "loss", where));
    std::unique_ptr<Optimizer> optimizer = build_optimizer(member(description, "optimizer", where));
    Training training(std::move(loss), std::move(optimizer), batch_size, build_random_state(text(description, "random_state", where), where));
    training.epoch = epoch;
    training.losses = losses;
    if (validation_losses)
        training.validation_losses = build_losses(*validation_losses, where + ".validation_losses");
    training.validation_split = number_or(description, "validation_split", where, training.validation_split);
    training.patience = count_or(description, "patience", where, training.patience);
    training.max_epochs = count_or(description, "max_epochs", where, training.max_epochs);
    restore_optimizer_memory(*training.optimizer, network, arrays);
    return training;
}

SavedNetwork load_network(const std::string &path)
{
    std::string bytes = read_file(path);
    try {
        Arrays arrays = read_archive(bytes);
        Json header = read_header(arrays);
        const Json *metadata = header.find("metadata");
        if (metadata && !metadata->is_object())
            throw Invalid("header.metadata must be an object");
        SavedNetwork saved {build_network(header, arrays), metadata ? *metadata : Json(Json::Object()), std::nullopt};
        if (const Json *training = header.find("training"))
            saved.training.emplace(build_training(*training, saved.network, arrays));
        return saved;
    } catch (const std::exception &error) {
        throw NetworkFileError("'" + path + "' is not a valid my_torch network: " + error.what());
    }
}

}
