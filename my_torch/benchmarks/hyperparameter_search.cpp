/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Benchmark of the hyperparameter search on the encoded chess dataset: "search" runs one particle swarm or
** genetic algorithm search scored by validation error; "curves" retrains, epoch by epoch, the best
** configuration found for each hidden activation and loss pair and logs its learning curves
*/

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/boards/boards.hpp"
#include "training/hyperparameters/algorithms/genetic_algorithm/genetic_algorithm.hpp"
#include "training/hyperparameters/algorithms/particle_swarm/particle_swarm.hpp"
#include "training/hyperparameters/search_log/search_log.hpp"
#include "training/hyperparameters/validation_loss/validation_loss.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"

static const int EXIT_ERROR = 84;

static const char *USAGE = "USAGE\n"
    "    ./hyperparameter_benchmark search pso|ga SEED ENCODED_DIR OUT_DIR EPOCHS\n"
    "    ./hyperparameter_benchmark curves ENCODED_DIR OUT_DIR EPOCHS\n";

static const char *CLASSES[] = {"Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"};

struct Dataset {
    my_torch::Boards train;
    my_torch::Boards validation;
    my_torch::Boards test;
};

struct Metrics {
    double loss;
    double accuracy;
    double macro_f1;
    std::vector<double> precision;
    std::vector<double> recall;
    std::vector<double> f1;
};

struct Candidate {
    std::string source;
    my_torch::Hyperparameters hyperparameters;
    double score;
};

static my_torch::Boards read_boards(const std::string &path)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("cannot open '" + path + "'");
    my_torch::BoardsReader reader(file);
    return reader.all();
}

static Dataset read_dataset(const std::string &folder)
{
    return {read_boards(folder + "/train.txt"), read_boards(folder + "/validation.txt"), read_boards(folder + "/test.txt")};
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
        my_torch::categorical("activation", my_torch::ValidationLoss::activation_choices()),
        my_torch::categorical("loss", my_torch::ValidationLoss::loss_choices()),
    });
}

static my_torch::SearchResult run_search(const std::string &method, unsigned int seed, const my_torch::ValidationLoss &score)
{
    std::mt19937 rng(seed);
    my_torch::ValidationError objective(score);
    if (method == "pso") {
        my_torch::SwarmSettings settings;
        settings.particles = 8;
        settings.iterations = 9;
        return my_torch::ParticleSwarm(search_space(), settings).minimize(objective, rng);
    }
    if (method != "ga")
        throw std::invalid_argument("unknown method '" + method + "', expected pso or ga");
    my_torch::GeneticSettings settings;
    settings.population = 8;
    settings.generations = 10;
    return my_torch::GeneticAlgorithm(search_space(), settings).minimize(objective, rng);
}

static std::vector<std::vector<double>> confusion(const my_torch::Matrix &predictions, const my_torch::Matrix &labels)
{
    std::vector<std::vector<double>> counts(labels.cols(), std::vector<double>(labels.cols(), 0.0));
    std::vector<std::size_t> predicted = predictions.row_argmax();
    std::vector<std::size_t> expected = labels.row_argmax();
    for (std::size_t row = 0; row < predicted.size(); row++)
        counts[expected[row]][predicted[row]] += 1.0;
    return counts;
}

static double ratio(double part, double whole)
{
    return whole > 0.0 ? part / whole : 0.0;
}

static Metrics measure(my_torch::Network &network, const my_torch::Boards &boards)
{
    my_torch::Matrix predictions = network(boards.inputs);
    std::vector<std::vector<double>> counts = confusion(predictions, boards.labels);
    Metrics metrics{my_torch::CrossEntropy()(predictions, boards.labels), my_torch::accuracy(predictions, boards.labels), 0.0, {}, {}, {}};
    for (std::size_t label = 0; label < counts.size(); label++) {
        double predicted = 0.0;
        double expected = 0.0;
        for (std::size_t other = 0; other < counts.size(); other++) {
            predicted += counts[other][label];
            expected += counts[label][other];
        }
        metrics.precision.push_back(ratio(counts[label][label], predicted));
        metrics.recall.push_back(ratio(counts[label][label], expected));
        metrics.f1.push_back(ratio(2.0 * metrics.precision.back() * metrics.recall.back(), metrics.precision.back() + metrics.recall.back()));
        metrics.macro_f1 += metrics.f1.back() / counts.size();
    }
    return metrics;
}

static std::string hyperparameters_text(const my_torch::Hyperparameters &values)
{
    std::string text;
    for (const auto &[name, value] : values) {
        if (name == "activation" || name == "loss")
            continue;
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%s%s=%.6g", text.empty() ? "" : " ", name.c_str(), value);
        text += buffer;
    }
    return text;
}

static int search(const std::vector<std::string> &arguments)
{
    if (arguments.size() != 6)
        throw std::invalid_argument(USAGE);
    unsigned int seed = static_cast<unsigned int>(std::stoul(arguments[2]));
    Dataset data = read_dataset(arguments[3]);
    my_torch::ValidationLoss score({data.train.inputs, data.train.labels, data.validation.inputs, data.validation.labels}, std::stoul(arguments[5]), seed);
    auto start = std::chrono::steady_clock::now();
    my_torch::SearchResult result = run_search(arguments[1], seed, score);
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::string prefix = arguments[4] + "/" + arguments[1] + "_seed" + arguments[2];
    my_torch::write_search_logs(result, prefix + "_evaluations.csv", prefix + "_convergence.csv");
    my_torch::Network network = score.train(result.best);
    Metrics test = measure(network, data.test);
    std::printf("%s,%u,%zu,%.6f,%.4f,%.6f,%.1f,%s,%s,%s\n", arguments[1].c_str(), seed, result.evaluations.size(), result.best_score, test.accuracy, test.loss, seconds, my_torch::ValidationLoss::activation_name(result.best).c_str(), my_torch::ValidationLoss::loss_name(result.best).c_str(), hyperparameters_text(result.best).c_str());
    return 0;
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

static void read_candidates(const std::filesystem::path &path, std::map<std::string, Candidate> &best)
{
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    std::vector<std::string> header = split_csv(line);
    std::string source = path.stem().string().substr(0, path.stem().string().find("_evaluations"));
    while (std::getline(file, line)) {
        std::vector<std::string> fields = split_csv(line);
        Candidate candidate{source, {}, std::stod(fields.back())};
        for (std::size_t column = 2; column + 1 < header.size(); column++)
            candidate.hyperparameters[header[column]] = std::stod(fields[column]);
        std::string pair = my_torch::ValidationLoss::activation_name(candidate.hyperparameters) + "_" + my_torch::ValidationLoss::loss_name(candidate.hyperparameters);
        if (!best.count(pair) || candidate.score < best[pair].score)
            best[pair] = candidate;
    }
}

static std::map<std::string, Candidate> best_per_pair(const std::string &folder)
{
    std::map<std::string, Candidate> best;
    for (const std::filesystem::directory_entry &entry : std::filesystem::directory_iterator(folder))
        if (entry.path().string().ends_with("_evaluations.csv"))
            read_candidates(entry.path(), best);
    if (best.empty())
        throw std::runtime_error("no *_evaluations.csv in '" + folder + "': run the searches first");
    return best;
}

static void write_classes(const std::string &path, const Metrics &test)
{
    std::ofstream file(path);
    file << "class,precision,recall,f1\n";
    for (std::size_t label = 0; label < test.f1.size(); label++)
        file << CLASSES[label] << "," << test.precision[label] << "," << test.recall[label] << "," << test.f1[label] << "\n";
}

static std::string curve_of(const std::string &pair, const Candidate &candidate, const Dataset &data, const std::string &folder, std::size_t epochs)
{
    my_torch::ValidationLoss score({data.train.inputs, data.train.labels, data.validation.inputs, data.validation.labels}, epochs, 0);
    std::mt19937 rng(0);
    my_torch::Network network = score.build(candidate.hyperparameters, rng);
    std::unique_ptr<my_torch::Loss> loss = score.loss(candidate.hyperparameters);
    std::unique_ptr<my_torch::Optimizer> optimizer = score.optimizer(candidate.hyperparameters);
    std::ofstream log(folder + "/curves_" + pair + ".csv");
    log << "epoch,train_loss,validation_cross_entropy,validation_accuracy,validation_macro_f1\n";
    Metrics best_test{};
    double best_accuracy = -1.0;
    std::size_t best_epoch = 0;
    for (std::size_t epoch = 1; epoch <= epochs; epoch++) {
        double train_loss = network.train(data.train.inputs, data.train.labels, *loss, *optimizer, 1, score.batch_size(candidate.hyperparameters), rng, epoch - 1)[0];
        Metrics validation = measure(network, data.validation);
        log << epoch << "," << train_loss << "," << validation.loss << "," << validation.accuracy << "," << validation.macro_f1 << "\n";
        if (validation.accuracy > best_accuracy) {
            best_accuracy = validation.accuracy;
            best_epoch = epoch;
            best_test = measure(network, data.test);
        }
    }
    write_classes(folder + "/curves_" + pair + "_classes.csv", best_test);
    char line[512];
    std::snprintf(line, sizeof(line), "%s,%s,%.6f,%zu,%.4f,%.4f,%.4f,%s\n", pair.c_str(), candidate.source.c_str(), candidate.score, best_epoch, best_accuracy, best_test.accuracy, best_test.macro_f1, hyperparameters_text(candidate.hyperparameters).c_str());
    return line;
}

static int curves(const std::vector<std::string> &arguments)
{
    if (arguments.size() != 4)
        throw std::invalid_argument(USAGE);
    Dataset data = read_dataset(arguments[1]);
    std::ofstream summary(arguments[2] + "/curves_summary.csv");
    summary << "pair,found_by,search_validation_error,best_epoch,validation_accuracy,test_accuracy,test_macro_f1,hyperparameters\n";
    for (const auto &[pair, candidate] : best_per_pair(arguments[2])) {
        std::string line = curve_of(pair, candidate, data, arguments[2], std::stoul(arguments[3]));
        summary << line;
        std::cout << line;
    }
    return 0;
}

int main(int argc, char **argv)
{
    std::vector<std::string> arguments(argv + 1, argv + argc);
    try {
        if (!arguments.empty() && arguments[0] == "search")
            return search(arguments);
        if (!arguments.empty() && arguments[0] == "curves")
            return curves(arguments);
        throw std::invalid_argument(USAGE);
    } catch (const std::exception &error) {
        std::cerr << "hyperparameter_benchmark: " << error.what() << std::endl;
        return EXIT_ERROR;
    }
}
