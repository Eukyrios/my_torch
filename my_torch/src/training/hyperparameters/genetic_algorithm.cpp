/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A child takes each gene somewhere around its two parents' (blend crossover, 25% beyond either
** side), then each gene may move by a Gaussian step; elites go to the next generation unchanged
*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "training/hyperparameters/genetic_algorithm.hpp"

namespace my_torch {

struct Individual {
    std::vector<double> genes;
    double score;
};

static const double BLEND = 0.25;

static void check_rate(double rate, const char *name)
{
    if (rate < 0.0 || rate > 1.0)
        throw std::invalid_argument(std::string("the ") + name + " must be in [0, 1], got " + std::to_string(rate));
}

static void check_settings(const GeneticSettings &settings)
{
    if (settings.population < 2)
        throw std::invalid_argument("a population needs at least 2 individuals");
    if (settings.elites >= settings.population)
        throw std::invalid_argument("the elites must leave room for at least one child per generation");
    if (settings.tournament < 1 || settings.tournament > settings.population)
        throw std::invalid_argument("the tournament size must be between 1 and the population");
    check_rate(settings.crossover_rate, "crossover rate");
    check_rate(settings.mutation_rate, "mutation rate");
    if (!(settings.mutation_scale > 0.0))
        throw std::invalid_argument("the mutation scale must be positive, got " + std::to_string(settings.mutation_scale));
}

static double usable_score(double score)
{
    return std::isfinite(score) ? score : std::numeric_limits<double>::infinity();
}

static bool scores_lower(const Individual &first, const Individual &second)
{
    return first.score < second.score;
}

static Individual evaluate(const std::vector<double> &genes, std::size_t generation, std::size_t index, const SearchSpace &space, const Objective &objective, SearchResult &result)
{
    Hyperparameters hyperparameters = space.decode(genes);
    double score = usable_score(objective(hyperparameters));
    result.evaluations.push_back({generation, index, hyperparameters, score});
    return {genes, score};
}

static const Individual &tournament(const std::vector<Individual> &population, std::size_t size, std::mt19937 &rng)
{
    std::uniform_int_distribution<std::size_t> pick(0, population.size() - 1);
    const Individual *winner = &population[pick(rng)];
    for (std::size_t round = 1; round < size; round++) {
        const Individual &challenger = population[pick(rng)];
        if (challenger.score < winner->score)
            winner = &challenger;
    }
    return *winner;
}

static std::vector<double> crossover(const std::vector<double> &first, const std::vector<double> &second, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> mix(-BLEND, 1.0 + BLEND);
    std::vector<double> child;
    for (std::size_t gene = 0; gene < first.size(); gene++)
        child.push_back(first[gene] + mix(rng) * (second[gene] - first[gene]));
    return child;
}

static void mutate(std::vector<double> &genes, const GeneticSettings &settings, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    std::normal_distribution<double> step(0.0, settings.mutation_scale);
    for (double &gene : genes) {
        if (chance(rng) < settings.mutation_rate)
            gene += step(rng);
        gene = std::clamp(gene, 0.0, 1.0);
    }
}

static std::vector<double> child_of(const std::vector<Individual> &population, const GeneticSettings &settings, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> chance(0.0, 1.0);
    const Individual &first = tournament(population, settings.tournament, rng);
    const Individual &second = tournament(population, settings.tournament, rng);
    std::vector<double> genes = chance(rng) < settings.crossover_rate ? crossover(first.genes, second.genes, rng) : first.genes;
    mutate(genes, settings, rng);
    return genes;
}

static std::vector<Individual> elites_of(std::vector<Individual> population, std::size_t count)
{
    std::stable_sort(population.begin(), population.end(), scores_lower);
    population.resize(count);
    return population;
}

GeneticAlgorithm::GeneticAlgorithm(const SearchSpace &space, const GeneticSettings &settings)
:
    space(space), settings(settings)
{
    check_settings(settings);
}

SearchResult GeneticAlgorithm::minimize(const Objective &objective, std::mt19937 &rng) const
{
    SearchResult result;
    std::vector<Individual> population;
    for (std::size_t index = 0; index < settings.population; index++)
        population.push_back(evaluate(space.random_position(rng), 0, index, space, objective, result));
    Individual best = *std::min_element(population.begin(), population.end(), scores_lower);
    result.best_per_iteration.push_back(best.score);
    for (std::size_t generation = 1; generation <= settings.generations; generation++) {
        std::vector<Individual> next = elites_of(population, settings.elites);
        while (next.size() < settings.population)
            next.push_back(evaluate(child_of(population, settings, rng), generation, next.size(), space, objective, result));
        population = next;
        best = std::min(best, *std::min_element(population.begin(), population.end(), scores_lower), scores_lower);
        result.best_per_iteration.push_back(best.score);
    }
    result.best = space.decode(best.genes);
    result.best_score = best.score;
    return result;
}

}
