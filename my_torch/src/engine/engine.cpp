/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** predict writes one line per board, its outputs separated by spaces, written with the fewest
** digits that read back as the same doubles; it works a chunk of boards at a time. train prints
** one line per epoch, keeps the best weights and saves the network with its whole training, so
** the next --train carries on from there. Errors go to err and give the exit code 84
*/

#include <charconv>
#include <iomanip>
#include <stdexcept>

#include "engine/engine.hpp"
#include "io/boards/boards.hpp"
#include "io/network_file/network_file.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/schedules/schedules/exponential_decay/exponential_decay.hpp"

namespace my_torch {

static const int EXIT_SUCCESS_CODE = 0;
static const int EXIT_ERROR_CODE = 84;

static const char *USAGE = "USAGE\n"
    "    my_torch_engine predict LOADFILE < BOARDS\n"
    "    my_torch_engine train LOADFILE SAVEFILE < BOARDS\n";

Training default_training()
{
    // the best configuration of the particle swarm search (docs/hyperparameter_search.md)
    Training training(std::make_unique<CrossEntropy>(), std::make_unique<Momentum>(0.018, 0.53, std::make_unique<ExponentialDecay>(0.86)), 9, std::mt19937(0));
    return training;
}

static void check_sizes(const Network &network, const BoardsReader &reader)
{
    if (reader.n_inputs != network.shape().first)
        throw BoardsError("the network takes " + std::to_string(network.shape().first) + " inputs, the boards have " + std::to_string(reader.n_inputs));
}

static std::string number(double value)
{
    char text[32];
    return std::string(text, std::to_chars(text, text + sizeof(text), value).ptr);
}

void predict(const std::string &network_path, std::istream &boards, std::ostream &out)
{
    SavedNetwork saved = load_network(network_path);
    BoardsReader reader(boards);
    check_sizes(saved.network, reader);
    for (Boards chunk = reader.next(PREDICT_CHUNK); chunk.inputs.rows() > 0; chunk = reader.next(PREDICT_CHUNK)) {
        Matrix outputs = saved.network(chunk.inputs);
        for (std::size_t row = 0; row < outputs.rows(); row++)
            for (std::size_t col = 0; col < outputs.cols(); col++)
                out << number(outputs(row, col)) << (col + 1 < outputs.cols() ? ' ' : '\n');
    }
}

static void check_training_boards(const Network &network, const BoardsReader &reader, const Boards &boards)
{
    if (reader.n_classes != network.shape().second)
        throw BoardsError("the network has " + std::to_string(network.shape().second) + " outputs, the boards have " + std::to_string(reader.n_classes) + " classes");
    if (boards.labelled != boards.inputs.rows())
        throw BoardsError("every board needs its class to train, " + std::to_string(boards.inputs.rows() - boards.labelled) + " have none");
}

static void print_epoch(std::ostream &out, const EpochReport &report)
{
    out << "epoch " << report.epoch + 1 << ": loss " << std::fixed << std::setprecision(4) << report.loss
        << ", validation loss " << report.validation_loss << (report.best ? " (best)" : "") << std::endl;
}

static void print_result(std::ostream &out, const FitResult &result, const std::string &save_path)
{
    out << std::fixed << std::setprecision(4);
    if (result.stopped_early)
        out << "stopped early: no better validation loss for " << result.epochs - result.best_epochs << " epochs" << std::endl;
    if (result.best_epochs == 0)
        out << "kept the weights loaded: validation loss " << result.initial_validation_loss << std::endl;
    else
        out << "kept the weights after epoch " << result.best_epochs << " of " << result.epochs << ": validation loss "
            << result.initial_validation_loss << " -> " << result.best_validation_loss << std::endl;
    out << "saved to " << save_path << std::endl;
}

void train(const std::string &network_path, const std::string &save_path, std::istream &boards, std::ostream &out)
{
    SavedNetwork saved = load_network(network_path);
    BoardsReader reader(boards);
    check_sizes(saved.network, reader);
    Boards data = reader.all();
    check_training_boards(saved.network, reader, data);
    Training training = saved.training ? std::move(*saved.training) : default_training();
    out << "training on " << data.inputs.rows() << " boards, " << validation_rows(data.inputs.rows(), training.validation_split).size()
        << " kept aside for validation, at most " << training.max_epochs << " epochs" << std::endl;
    FitResult result = training.fit(saved.network, data.inputs, data.labels, [&](const EpochReport &report) { print_epoch(out, report); });
    save_network(save_path, saved.network, saved.metadata, &training);
    print_result(out, result, save_path);
}

int run_engine(const std::vector<std::string> &arguments, std::istream &in, std::ostream &out, std::ostream &err)
{
    try {
        if (arguments.size() == 2 && arguments[0] == "predict")
            predict(arguments[1], in, out);
        else if (arguments.size() == 3 && arguments[0] == "train")
            train(arguments[1], arguments[2], in, out);
        else
            throw std::invalid_argument(USAGE);
    } catch (const std::exception &error) {
        err << error.what() << std::endl;
        return EXIT_ERROR_CODE;
    }
    return EXIT_SUCCESS_CODE;
}

}
