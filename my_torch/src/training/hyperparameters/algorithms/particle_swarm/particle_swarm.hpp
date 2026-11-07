/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Particle swarm optimization: each particle is a configuration that moves through the search
** space, pulled towards its own best position and the swarm's; the first particles can start from
** known good positions; minimize returns every evaluation
*/

#pragma once

#include <cstddef>
#include <random>

#include "training/hyperparameters/search_result/search_result.hpp"
#include "training/hyperparameters/search_space/search_space.hpp"

namespace my_torch {

struct SwarmSettings {
    std::size_t particles = 10;
    std::size_t iterations = 20;
    double inertia = 0.7;
    double cognitive = 1.5;
    double social = 1.5;
    double max_velocity = 0.2;
};

class ParticleSwarm {
public:
    explicit ParticleSwarm(const SearchSpace &space, const SwarmSettings &settings = SwarmSettings());

    SearchResult minimize(const Objective &objective, std::mt19937 &rng, const std::vector<std::vector<double>> &starts = {}) const;

    SearchSpace space;
    SwarmSettings settings;
};

}
