/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the CSV logs of a particle swarm search
*/

#include <algorithm>
#include <criterion/criterion.h>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

#include "training/hyperparameters/particle_swarm.hpp"
#include "training/hyperparameters/search_log.hpp"

static my_torch::SearchResult small_result()
{
    my_torch::SearchResult result;
    result.evaluations.push_back({0, 0, {{"momentum", 0.9}, {"learning_rate", 0.01}}, 0.5});
    result.evaluations.push_back({0, 1, {{"momentum", 0.5}, {"learning_rate", 0.1}}, 0.25});
    result.evaluations.push_back({1, 0, {{"momentum", 0.7}, {"learning_rate", 0.05}}, 0.125});
    result.best_per_iteration = {0.25, 0.125};
    result.best = result.evaluations.back().hyperparameters;
    result.best_score = 0.125;
    return result;
}

static double sphere(const my_torch::Hyperparameters &values)
{
    return values.at("x") * values.at("x");
}

static std::string read_text(const std::string &path)
{
    std::ifstream file(path);
    std::stringstream text;
    text << file.rdbuf();
    return text.str();
}

Test(search_log, lists_every_evaluation_with_its_hyperparameters_in_name_order)
{
    std::string text = my_torch::evaluations_csv(small_result());

    cr_assert(text == "iteration,particle,learning_rate,momentum,score\n0,0,0.01,0.90000000000000002,0.5\n0,1,0.10000000000000001,0.5,0.25\n1,0,0.050000000000000003,0.69999999999999996,0.125\n", "%s", text.c_str());
}

Test(search_log, gives_the_best_score_after_each_iteration)
{
    std::string text = my_torch::convergence_csv(small_result());

    cr_assert(text == "iteration,best_score\n0,0.25\n1,0.125\n", "%s", text.c_str());
}

Test(search_log, logged_scores_read_back_as_the_same_double)
{
    my_torch::SearchResult result = small_result();
    result.best_per_iteration = {1.0 / 3.0};
    std::string text = my_torch::convergence_csv(result);

    cr_assert(std::stod(text.substr(text.find("0,") + 2)) == 1.0 / 3.0);
}

Test(search_log, has_one_line_per_evaluation_of_a_real_search)
{
    std::mt19937 rng(0);
    my_torch::SwarmSettings settings;
    settings.particles = 3;
    settings.iterations = 2;
    my_torch::SearchResult result = my_torch::ParticleSwarm(my_torch::SearchSpace({{"x", -1, 1}}), settings).minimize(sphere, rng);
    std::string text = my_torch::evaluations_csv(result);

    cr_assert_eq(std::count(text.begin(), text.end(), '\n'), 1 + 3 * 3);
}

Test(search_log, writes_both_files)
{
    std::string evaluations = "/tmp/my_torch_test_" + std::to_string(getpid()) + "_evaluations.csv";
    std::string convergence = "/tmp/my_torch_test_" + std::to_string(getpid()) + "_convergence.csv";

    my_torch::write_search_logs(small_result(), evaluations, convergence);
    cr_assert(read_text(evaluations) == my_torch::evaluations_csv(small_result()));
    cr_assert(read_text(convergence) == my_torch::convergence_csv(small_result()));
    std::remove(evaluations.c_str());
    std::remove(convergence.c_str());
}

Test(search_log, refuses_a_path_it_cannot_write)
{
    cr_assert_throw(my_torch::write_search_logs(small_result(), "/no/such/folder/evaluations.csv", "/tmp/convergence.csv"), std::runtime_error);
}
