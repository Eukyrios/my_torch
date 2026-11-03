/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** What every hyperparameter search shares: the objective to minimize and its result, the best
** configuration found plus every evaluation and the best score after each iteration
*/

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "training/hyperparameters/search_space.hpp"

namespace my_torch {

using Objective = std::function<double(const Hyperparameters &)>;

struct Evaluation {
    std::size_t iteration;
    std::size_t candidate;
    Hyperparameters hyperparameters;
    double score;
};

struct SearchResult {
    Hyperparameters best;
    double best_score;
    std::vector<Evaluation> evaluations;
    std::vector<double> best_per_iteration;
};

}
