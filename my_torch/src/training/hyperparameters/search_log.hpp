/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** CSV logs of a particle swarm search for the benchmark docs: every evaluation with its
** hyperparameters and score, and the best score after each iteration (the convergence curve)
*/

#pragma once

#include <string>

#include "training/hyperparameters/particle_swarm.hpp"

namespace my_torch {

std::string evaluations_csv(const SwarmResult &result);
std::string convergence_csv(const SwarmResult &result);
void write_search_logs(const SwarmResult &result, const std::string &evaluations_path, const std::string &convergence_path);

}
