/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Particle swarm optimization: each particle is a configuration that moves through the search
** space, pulled towards its own best position and the swarm's; minimize returns every evaluation
*/

#pragma once

#include <cstddef>
#include <functional>
#include <random>
#include <vector>

#include "training/hyperparameters/search_space.hpp"

namespace my_torch {

using Objective = std::function<double(const Hyperparameters &)>;

struct SwarmSettings {
    std::size_t particles = 10;
    std::size_t iterations = 20;
    double inertia = 0.7;
    double cognitive = 1.5;
    double social = 1.5;
    double max_velocity = 0.2;
};

struct Evaluation {
    std::size_t iteration;
    std::size_t particle;
    Hyperparameters hyperparameters;
    double score;
};

struct SwarmResult {
    Hyperparameters best;
    double best_score;
    std::vector<Evaluation> evaluations;
    std::vector<double> best_per_iteration;
};

class ParticleSwarm {
public:
    explicit ParticleSwarm(const SearchSpace &space, const SwarmSettings &settings = SwarmSettings());

    SwarmResult minimize(const Objective &objective, std::mt19937 &rng) const;

    SearchSpace space;
    SwarmSettings settings;
};

}
