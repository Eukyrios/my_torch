/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Genetic algorithm: a population of configurations evolves by tournament selection, blend
** crossover and Gaussian mutation, keeping its elites; minimize returns every evaluation
*/

#pragma once

#include <cstddef>
#include <random>

#include "training/hyperparameters/search_result.hpp"
#include "training/hyperparameters/search_space.hpp"

namespace my_torch {

struct GeneticSettings {
    std::size_t population = 10;
    std::size_t generations = 20;
    std::size_t elites = 1;
    std::size_t tournament = 3;
    double crossover_rate = 0.9;
    double mutation_rate = 0.2;
    double mutation_scale = 0.1;
};

class GeneticAlgorithm {
public:
    explicit GeneticAlgorithm(const SearchSpace &space, const GeneticSettings &settings = GeneticSettings());

    SearchResult minimize(const Objective &objective, std::mt19937 &rng) const;

    SearchSpace space;
    GeneticSettings settings;
};

}
