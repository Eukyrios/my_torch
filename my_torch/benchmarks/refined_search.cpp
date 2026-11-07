/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Refined search: a particle swarm over ReLU + cross-entropy networks trained like --train, its ranges taken
** from the best configurations of the first benchmark (and the current default), which it starts from,
** plus dropout and L2
*/

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/boards/boards.hpp"
#include "training/hyperparameters/algorithms/particle_swarm/particle_swarm.hpp"
#include "training/hyperparameters/fit_score/fit_score.hpp"
#include "training/hyperparameters/search_log/search_log.hpp"

static const int EXIT_ERROR = 84;

static const char *USAGE = "USAGE\n    ./refined_benchmark SEED ENCODED_DIR FIRST_RESULTS_DIR OUT_DIR TRAIN_BOARDS MAX_EPOCHS\n";

static const double TOP_SHARE = 0.25;
static const std::size_t PARTICLES = 8;
static const std::size_t ITERATIONS = 7;
static const std::size_t FULL_SCALE_EPOCHS = 50;

struct Earlier {
    my_torch::Hyperparameters hyperparameters;
    double score;
};

static const std::vector<my_torch::Dimension> &first_dimensions()
{
    static const std::vector<my_torch::Dimension> dimensions = {
        {"learning_rate", 1e-4, 0.3, true, false},
        {"momentum", 0.0, 0.99, false, false},
        {"batch_size", 8, 128, true, true},
        {"hidden_size", 16, 256, true, true},
        {"hidden_layers", 1, 3, false, true},
        {"decay", 0.8, 1.0, false, false},
    };
    return dimensions;
}

static my_torch::Boards read_boards(const std::string &path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open '" + path + "'");
    my_torch::BoardsReader reader(file);
    return reader.all();
}

static my_torch::Boards sample(const my_torch::Boards &boards, std::size_t count)
{
    if (count == 0 || count >= boards.inputs.rows())
        return boards;
    std::vector<std::size_t> rows(boards.inputs.rows());
    std::iota(rows.begin(), rows.end(), 0);
    std::mt19937 rng(0);
    std::shuffle(rows.begin(), rows.end(), rng);
    rows.resize(count);
    return {boards.inputs.select_rows(rows), boards.labels.select_rows(rows), count};
}

static std::vector<std::string> split_csv(const std::string &line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ','))
        fields.push_back(field);
    return fields;
}

static bool is_relu_cross_entropy(const my_torch::Hyperparameters &values)
{
    return (!values.count("activation") || values.at("activation") == 0.0) && (!values.count("loss") || values.at("loss") == 0.0);
}

static void read_earlier(const std::filesystem::path &path, std::vector<Earlier> &earlier)
{
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    std::vector<std::string> header = split_csv(line);
    while (std::getline(file, line)) {
        std::vector<std::string> fields = split_csv(line);
        Earlier configuration{{}, std::stod(fields.back())};
        for (std::size_t column = 2; column + 1 < header.size(); column++)
            configuration.hyperparameters[header[column]] = std::stod(fields[column]);
        if (is_relu_cross_entropy(configuration.hyperparameters))
            earlier.push_back(configuration);
    }
}

static bool scores_lower(const Earlier &first, const Earlier &second)
{
    return first.score < second.score;
}

static std::vector<Earlier> best_earlier(const std::string &folder)
{
    std::vector<Earlier> earlier;
    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(folder))
        if (entry.path().string().ends_with("_evaluations.csv"))
            read_earlier(entry.path(), earlier);
    if (earlier.empty())
        throw std::runtime_error("no ReLU + cross-entropy configuration in the *_evaluations.csv of '" + folder + "'");
    std::sort(earlier.begin(), earlier.end(), scores_lower);
    earlier.resize(std::max<std::size_t>(PARTICLES, static_cast<std::size_t>(earlier.size() * TOP_SHARE)));
    return earlier;
}

static my_torch::Dimension narrowed(const my_torch::Dimension &dimension, const std::vector<Earlier> &best)
{
    double low = my_torch::FitScore::defaults().at(dimension.name);
    double high = low;
    for (const Earlier &configuration : best) {
        low = std::min(low, configuration.hyperparameters.at(dimension.name));
        high = std::max(high, configuration.hyperparameters.at(dimension.name));
    }
    my_torch::Dimension result = dimension;
    double margin = dimension.log_scale ? 1.25 : 0.05 * (dimension.high - dimension.low);
    result.low = std::max(dimension.low, dimension.log_scale ? low / margin : low - margin);
    result.high = std::min(dimension.high, dimension.log_scale ? high * margin : high + margin);
    if (dimension.integer) {
        result.low = std::floor(result.low);
        result.high = std::ceil(result.high);
    }
    if (!(result.low < result.high))
        result.high = std::min(dimension.high, result.low + 1.0);
    return result;
}

static my_torch::SearchSpace refined_space(const std::vector<Earlier> &best)
{
    std::vector<my_torch::Dimension> dimensions;
    for (const my_torch::Dimension &dimension : first_dimensions())
        dimensions.push_back(narrowed(dimension, best));
    dimensions.push_back({"dropout", 0.0, 0.5, false, false});
    dimensions.push_back({"weight_decay", 1e-6, 1e-2, true, false});
    return my_torch::SearchSpace(dimensions);
}

static my_torch::Hyperparameters as_start(const my_torch::Hyperparameters &earlier)
{
    my_torch::Hyperparameters values;
    for (const my_torch::Dimension &dimension : first_dimensions())
        values[dimension.name] = earlier.at(dimension.name);
    values["dropout"] = 0.0;
    values["weight_decay"] = 1e-6;
    return values;
}

static std::vector<std::vector<double>> starts(const my_torch::SearchSpace &space, const std::vector<Earlier> &best)
{
    std::vector<std::vector<double>> positions = {space.encode(as_start(my_torch::FitScore::defaults()))};
    for (std::size_t index = 0; index < best.size() && positions.size() < PARTICLES / 2; index++)
        positions.push_back(space.encode(as_start(best[index].hyperparameters)));
    return positions;
}

static void write_space(const std::string &path, const my_torch::SearchSpace &space)
{
    std::ofstream file(path);
    file << "hyperparameter,low,high,scale\n";
    for (const my_torch::Dimension &dimension : space.dimensions)
        file << dimension.name << "," << dimension.low << "," << dimension.high << "," << (dimension.log_scale ? "log" : "linear") << (dimension.integer ? " whole" : "") << "\n";
}

static void write_config(const std::string &path, const my_torch::FitScore &score, const my_torch::Hyperparameters &best, unsigned int seed)
{
    my_torch::Json config = score.config(best, FULL_SCALE_EPOCHS);
    config.set("metadata", my_torch::Json(my_torch::Json::Object {{"description", "best configuration of the refined particle swarm search, seed " + std::to_string(seed) + " (docs/refined_search.md)"}}));
    std::ofstream(path) << config.dump() << "\n";
}

static std::string hyperparameters_text(const my_torch::Hyperparameters &values)
{
    std::string text;
    for (const auto &[name, value] : values) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%s%s=%.6g", text.empty() ? "" : " ", name.c_str(), value);
        text += buffer;
    }
    return text;
}

static int run(const std::vector<std::string> &arguments)
{
    if (arguments.size() != 6)
        throw std::invalid_argument(USAGE);
    unsigned int seed = static_cast<unsigned int>(std::stoul(arguments[0]));
    my_torch::Boards train = sample(read_boards(arguments[1] + "/train.txt"), std::stoul(arguments[4]));
    my_torch::Boards validation = read_boards(arguments[1] + "/validation.txt");
    std::vector<Earlier> best = best_earlier(arguments[2]);
    my_torch::SearchSpace space = refined_space(best);
    my_torch::FitScore score({train.inputs, train.labels, validation.inputs, validation.labels}, std::stoul(arguments[5]), seed);
    my_torch::SwarmSettings settings;
    settings.particles = PARTICLES;
    settings.iterations = ITERATIONS;
    std::mt19937 rng(seed);
    auto start = std::chrono::steady_clock::now();
    my_torch::SearchResult result = my_torch::ParticleSwarm(space, settings).minimize(score, rng, starts(space, best));
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::string prefix = arguments[3] + "/refined_seed" + arguments[0];
    my_torch::write_search_logs(result, prefix + "_evaluations.csv", prefix + "_convergence.csv");
    write_space(prefix + "_space.csv", space);
    write_config(prefix + "_best.conf", score, result.best, seed);
    std::printf("refined,%u,%zu,%zu,%.6f,%.6f,%.1f,%s\n", seed, train.inputs.rows(), result.evaluations.size(), result.best_score, result.evaluations.front().score, seconds, hyperparameters_text(result.best).c_str());
    return 0;
}

int main(int argc, char **argv)
{
    try {
        return run(std::vector<std::string>(argv + 1, argv + argc));
    } catch (const std::exception &error) {
        std::cerr << "refined_benchmark: " << error.what() << std::endl;
        return EXIT_ERROR;
    }
}
