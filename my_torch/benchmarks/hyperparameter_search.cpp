/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Benchmark: one particle swarm or genetic algorithm search on the encoded chess dataset; writes
** the CSV logs, retrains the best configuration and prints its validation and test results
*/

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "training/hyperparameters/algorithms/genetic_algorithm/genetic_algorithm.hpp"
#include "training/hyperparameters/algorithms/particle_swarm/particle_swarm.hpp"
#include "training/hyperparameters/search_log/search_log.hpp"
#include "training/hyperparameters/validation_loss/validation_loss.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"

static const int EXIT_ERROR = 84;

static const char *USAGE = "USAGE\n    ./hyperparameter_benchmark pso|ga SEED ENCODED_DIR OUT_DIR EPOCHS\n";

struct Split {
    my_torch::Matrix inputs;
    my_torch::Matrix labels;
};

struct Options {
    std::string method;
    unsigned int seed;
    std::string data;
    std::string out;
    std::size_t epochs;
};

static Options parse_options(int argc, char **argv)
{
    if (argc != 6)
        throw std::invalid_argument(USAGE);
    Options options{argv[1], static_cast<unsigned int>(std::stoul(argv[2])), argv[3], argv[4], std::stoul(argv[5])};
    if (options.method != "pso" && options.method != "ga")
        throw std::invalid_argument(std::string("unknown method '") + argv[1] + "', expected pso or ga");
    return options;
}

static void read_board(const std::string &line, std::size_t row, Split &split)
{
    std::istringstream fields(line);
    std::size_t label = 0;
    std::size_t index = 0;
    fields >> label;
    split.labels(row, label) = 1.0;
    while (fields >> index)
        split.inputs(row, index) = 1.0;
}

static Split read_split(const std::string &path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open '" + path + "'");
    std::size_t inputs = 0;
    std::size_t classes = 0;
    file >> inputs >> classes;
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line))
        if (!line.empty())
            lines.push_back(line);
    Split split{my_torch::Matrix(lines.size(), inputs), my_torch::Matrix(lines.size(), classes)};
    for (std::size_t row = 0; row < lines.size(); row++)
        read_board(lines[row], row, split);
    return split;
}

static my_torch::SearchSpace search_space()
{
    return my_torch::SearchSpace({
        {"learning_rate", 1e-4, 0.3, true, false},
        {"momentum", 0.0, 0.99, false, false},
        {"batch_size", 8, 128, true, true},
        {"hidden_size", 16, 256, true, true},
        {"hidden_layers", 1, 3, false, true},
        {"decay", 0.8, 1.0, false, false},
    });
}

static my_torch::SearchResult run_search(const Options &options, const my_torch::ValidationLoss &score)
{
    std::mt19937 rng(options.seed);
    if (options.method == "pso") {
        my_torch::SwarmSettings settings;
        settings.particles = 8;
        settings.iterations = 9;
        return my_torch::ParticleSwarm(search_space(), settings).minimize(score, rng);
    }
    my_torch::GeneticSettings settings;
    settings.population = 8;
    settings.generations = 10;
    return my_torch::GeneticAlgorithm(search_space(), settings).minimize(score, rng);
}

static double accuracy(const my_torch::Matrix &predictions, const my_torch::Matrix &labels)
{
    std::vector<std::size_t> predicted = predictions.row_argmax();
    std::vector<std::size_t> expected = labels.row_argmax();
    std::size_t right = 0;
    for (std::size_t row = 0; row < predicted.size(); row++)
        right += predicted[row] == expected[row];
    return static_cast<double>(right) / predicted.size();
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

static void report(const Options &options, const my_torch::SearchResult &result, const my_torch::ValidationLoss &score, const Split &test, double seconds)
{
    my_torch::Network network = score.train(result.best);
    my_torch::Matrix predictions = network(test.inputs);
    std::printf("%s,%u,%zu,%.6f,%.6f,%.4f,%.1f,%s\n", options.method.c_str(), options.seed, result.evaluations.size(), result.best_score, my_torch::CrossEntropy()(predictions, test.labels), accuracy(predictions, test.labels), seconds, hyperparameters_text(result.best).c_str());
}

static int run(const Options &options)
{
    Split train = read_split(options.data + "/train.txt");
    Split validation = read_split(options.data + "/validation.txt");
    Split test = read_split(options.data + "/test.txt");
    my_torch::ValidationLoss score({train.inputs, train.labels, validation.inputs, validation.labels}, options.epochs, options.seed);
    auto start = std::chrono::steady_clock::now();
    my_torch::SearchResult result = run_search(options, score);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::string prefix = options.out + "/" + options.method + "_seed" + std::to_string(options.seed);
    my_torch::write_search_logs(result, prefix + "_evaluations.csv", prefix + "_convergence.csv");
    report(options, result, score, test, seconds);
    return 0;
}

int main(int argc, char **argv)
{
    try {
        return run(parse_options(argc, argv));
    } catch (const std::exception &error) {
        std::cerr << "hyperparameter_benchmark: " << error.what() << std::endl;
        return EXIT_ERROR;
    }
}
