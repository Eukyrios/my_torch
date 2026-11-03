/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Numbers are written with 17 significant digits so a logged score reads back as the same
** double; hyperparameter columns follow the alphabetical order of their names
*/

#include <cstdio>
#include <fstream>
#include <stdexcept>

#include "training/hyperparameters/search_log.hpp"

namespace my_torch {

static std::string number_text(double value)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.17g", value);
    return buffer;
}

static std::string header_line(const Hyperparameters &names)
{
    std::string line = "iteration,particle";
    for (const auto &[name, value] : names)
        line += "," + name;
    return line + ",score\n";
}

static std::string evaluation_line(const Evaluation &evaluation)
{
    std::string line = std::to_string(evaluation.iteration) + "," + std::to_string(evaluation.particle);
    for (const auto &[name, value] : evaluation.hyperparameters)
        line += "," + number_text(value);
    return line + "," + number_text(evaluation.score) + "\n";
}

static void write_text(const std::string &path, const std::string &text)
{
    std::ofstream file(path, std::ios::trunc);
    if (!file || !(file << text))
        throw std::runtime_error("cannot write '" + path + "'");
}

std::string evaluations_csv(const SearchResult &result)
{
    if (result.evaluations.empty())
        return "iteration,particle,score\n";
    std::string text = header_line(result.evaluations.front().hyperparameters);
    for (const Evaluation &evaluation : result.evaluations)
        text += evaluation_line(evaluation);
    return text;
}

std::string convergence_csv(const SearchResult &result)
{
    std::string text = "iteration,best_score\n";
    for (std::size_t iteration = 0; iteration < result.best_per_iteration.size(); iteration++)
        text += std::to_string(iteration) + "," + number_text(result.best_per_iteration[iteration]) + "\n";
    return text;
}

void write_search_logs(const SearchResult &result, const std::string &evaluations_path, const std::string &convergence_path)
{
    write_text(evaluations_path, evaluations_csv(result));
    write_text(convergence_path, convergence_csv(result));
}

}
