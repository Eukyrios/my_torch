/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the engine run by the analyzer: predict, train and their errors
*/

#include <criterion/criterion.h>
#include <algorithm>
#include <filesystem>
#include <iterator>
#include <sstream>
#include <unistd.h>

#include "engine/engine.hpp"
#include "io/network_file/network_file.hpp"
#include "model/activations/activations/softmax/softmax.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "training/optimizers/optimizers/momentum/momentum.hpp"
#include "training/schedules/schedules/exponential_decay/exponential_decay.hpp"

static std::string temporary_path(const std::string &name)
{
    std::filesystem::path directory = std::filesystem::temp_directory_path() / ("my_torch_engine_tests_" + std::to_string(getpid()));
    std::filesystem::create_directories(directory);
    return (directory / name).string();
}

// 3 inputs, 2 classes: input 0 says class 0, input 1 says class 1
static std::string saved_network(const std::string &name)
{
    std::vector<std::unique_ptr<my_torch::Layer>> layers;
    layers.push_back(std::make_unique<my_torch::DenseLayer>(my_torch::Matrix({{2, -2}, {-2, 2}, {0, 0}}), std::vector<double> {0, 0}, std::make_unique<my_torch::Softmax>()));
    my_torch::Network network(std::move(layers));
    std::string path = temporary_path(name);
    my_torch::save_network(path, network, my_torch::Json::Object {{"labels", my_torch::Json::Array {"a", "b"}}});
    return path;
}

struct Run {
    int code;
    std::string out;
    std::string err;
};

static Run run(const std::vector<std::string> &arguments, const std::string &input)
{
    std::istringstream in(input);
    std::ostringstream out;
    std::ostringstream err;
    int code = my_torch::run_engine(arguments, in, out, err);
    return Run {code, out.str(), err.str()};
}

static std::string many_boards(std::size_t count)
{
    std::string text = "3 2\n";
    for (std::size_t board = 0; board < count; board++)
        text += board % 2 ? "1 1 2\n" : "0 0 2\n";
    return text;
}

Test(engine, the_default_training_is_the_one_the_search_retained)
{
    my_torch::Training training = my_torch::default_training();
    my_torch::Momentum &momentum = dynamic_cast<my_torch::Momentum &>(*training.optimizer);

    cr_assert_eq(momentum.initial_learning_rate, 0.018);
    cr_assert_eq(momentum.momentum, 0.53);
    cr_assert_eq(dynamic_cast<my_torch::ExponentialDecay &>(*momentum.schedule).factor, 0.86);
    cr_assert_eq(training.batch_size, 9);
    cr_assert_str_eq(training.loss->name.c_str(), "cross_entropy");
}

Test(engine, predict_prints_the_outputs_of_every_board_in_order)
{
    std::string path = saved_network("predict.nn");
    Run result = run({"predict", path}, "3 2\n- 0\n- 1\n\n- 2\n");

    cr_assert_eq(result.code, 0);
    std::istringstream lines(result.out);
    std::vector<std::vector<double>> rows;
    for (std::string line; std::getline(lines, line);) {
        std::istringstream values(line);
        rows.emplace_back(std::istream_iterator<double>(values), std::istream_iterator<double>());
    }
    cr_assert_eq(rows.size(), 3);
    cr_assert_gt(rows[0][0], rows[0][1]);
    cr_assert_lt(rows[1][0], rows[1][1]);
    cr_assert_eq(rows[2][0], 0.5);
    cr_assert_float_eq(rows[0][0] + rows[0][1], 1.0, 1e-12);
}

Test(engine, predict_goes_through_big_files_a_chunk_at_a_time)
{
    std::string path = saved_network("chunks.nn");
    Run result = run({"predict", path}, many_boards(my_torch::PREDICT_CHUNK * 2 + 3));

    cr_assert_eq(result.code, 0);
    cr_assert_eq(std::count(result.out.begin(), result.out.end(), '\n'), my_torch::PREDICT_CHUNK * 2 + 3);
}

Test(engine, train_saves_a_network_that_learned_and_its_whole_training)
{
    std::string path = saved_network("to_train.nn");
    std::string saved = temporary_path("trained.nn");
    Run result = run({"train", path, saved}, many_boards(60));

    cr_assert_eq(result.code, 0, "%s", result.err.c_str());
    cr_assert_neq(result.out.find("training on 60 boards, 6 kept aside for validation, at most 50 epochs"), std::string::npos);
    cr_assert_neq(result.out.find("epoch 1: loss "), std::string::npos);
    cr_assert_neq(result.out.find("saved to " + saved), std::string::npos);
    my_torch::SavedNetwork trained = my_torch::load_network(saved);
    cr_assert(trained.training.has_value());
    cr_assert_gt(trained.training->epoch, 0);
    cr_assert_eq(trained.training->batch_size, 9);
    cr_assert_eq(trained.metadata.find("labels")->as_array().size(), 2);
    cr_assert_lt(trained.training->validation_losses.front(), 0.2);
}

Test(engine, a_second_train_carries_on_from_the_saved_training)
{
    std::string path = saved_network("twice.nn");

    cr_assert_eq(run({"train", path, path}, many_boards(30)).code, 0);
    std::size_t first = my_torch::load_network(path).training->epoch;
    cr_assert_eq(run({"train", path, path}, many_boards(30)).code, 0);
    cr_assert_gt(my_torch::load_network(path).training->epoch, first);
}

Test(engine, reports_errors_with_exit_code_84)
{
    std::string path = saved_network("errors.nn");
    Run usage = run({"predict"}, "");
    Run missing = run({"predict", temporary_path("missing.nn")}, "3 2\n");
    Run inputs = run({"predict", path}, "769 2\n- 0\n");
    Run classes = run({"train", path, path}, "3 5\n0 0\n");
    Run unlabelled = run({"train", path, path}, "3 2\n0 0\n- 1\n");
    Run bad_line = run({"predict", path}, "3 2\n- 7\n");

    cr_assert_eq(usage.code, 84);
    cr_assert_neq(usage.err.find("USAGE"), std::string::npos);
    cr_assert_eq(missing.code, 84);
    cr_assert_neq(missing.err.find("cannot open the network file"), std::string::npos);
    cr_assert_neq(inputs.err.find("the network takes 3 inputs, the boards have 769"), std::string::npos);
    cr_assert_neq(classes.err.find("the network has 2 outputs, the boards have 5 classes"), std::string::npos);
    cr_assert_neq(unlabelled.err.find("every board needs its class to train, 1 have none"), std::string::npos);
    cr_assert_neq(bad_line.err.find("boards line 2: input 7 is not below 3"), std::string::npos);
    cr_assert(bad_line.out.empty());
}
