/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the particle swarm on functions whose minimum is known
*/

#include <algorithm>
#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "training/hyperparameters/particle_swarm.hpp"

static const my_torch::SearchSpace PLANE({{"x", -5.0, 5.0}, {"y", -5.0, 5.0}});

static double sphere(const my_torch::Hyperparameters &values)
{
    return std::pow(values.at("x") - 1.5, 2) + std::pow(values.at("y") + 2.0, 2);
}

static double rosenbrock(const my_torch::Hyperparameters &values)
{
    double x = values.at("x");
    double y = values.at("y");
    return std::pow(1.0 - x, 2) + 100.0 * std::pow(y - x * x, 2);
}

static double nan_on_the_right(const my_torch::Hyperparameters &values)
{
    return values.at("x") > 0.0 ? std::nan("") : -values.at("x");
}

static double distance_to_seven(const my_torch::Hyperparameters &values)
{
    return std::abs(values.at("hidden") - 7.0);
}

static bool scores_lower(const my_torch::Evaluation &first, const my_torch::Evaluation &second)
{
    return first.score < second.score;
}

static my_torch::SwarmSettings settings(std::size_t particles, std::size_t iterations)
{
    my_torch::SwarmSettings chosen;
    chosen.particles = particles;
    chosen.iterations = iterations;
    return chosen;
}

Test(particle_swarm, finds_the_minimum_of_a_shifted_sphere)
{
    std::mt19937 rng(0);
    my_torch::SearchResult result = my_torch::ParticleSwarm(PLANE, settings(15, 40)).minimize(sphere, rng);

    cr_assert_float_eq(result.best.at("x"), 1.5, 0.05);
    cr_assert_float_eq(result.best.at("y"), -2.0, 0.05);
    cr_assert_lt(result.best_score, 1e-3);
}

Test(particle_swarm, gets_close_to_the_minimum_of_rosenbrock)
{
    std::mt19937 rng(1);
    my_torch::SearchSpace valley({{"x", -2.0, 2.0}, {"y", -1.0, 3.0}});
    my_torch::SearchResult result = my_torch::ParticleSwarm(valley, settings(30, 100)).minimize(rosenbrock, rng);

    cr_assert_lt(result.best_score, 1e-2);
}

Test(particle_swarm, evaluates_every_particle_at_every_iteration)
{
    std::mt19937 rng(2);
    my_torch::SearchResult result = my_torch::ParticleSwarm(PLANE, settings(4, 6)).minimize(sphere, rng);

    cr_assert_eq(result.evaluations.size(), 4u * 7u);
    cr_assert_eq(result.best_per_iteration.size(), 7u);
    cr_assert_eq(result.evaluations.front().iteration, 0u);
    cr_assert_eq(result.evaluations.back().iteration, 6u);
    cr_assert_eq(result.evaluations.back().candidate, 3u);
}

Test(particle_swarm, the_best_score_never_gets_worse)
{
    std::mt19937 rng(3);
    my_torch::SearchResult result = my_torch::ParticleSwarm(PLANE, settings(5, 20)).minimize(sphere, rng);

    for (std::size_t i = 1; i < result.best_per_iteration.size(); i++)
        cr_assert_leq(result.best_per_iteration[i], result.best_per_iteration[i - 1]);
}

Test(particle_swarm, the_best_is_the_lowest_evaluation)
{
    std::mt19937 rng(4);
    my_torch::SearchResult result = my_torch::ParticleSwarm(PLANE, settings(5, 10)).minimize(sphere, rng);
    auto lowest = std::min_element(result.evaluations.begin(), result.evaluations.end(), scores_lower);

    cr_assert_float_eq(result.best_score, lowest->score, 1e-15);
    cr_assert(result.best == lowest->hyperparameters);
}

Test(particle_swarm, is_reproducible_from_the_same_seed)
{
    std::mt19937 first(5);
    std::mt19937 second(5);
    my_torch::ParticleSwarm swarm(PLANE, settings(6, 8));

    cr_assert(swarm.minimize(sphere, first).best_per_iteration == swarm.minimize(sphere, second).best_per_iteration);
}

Test(particle_swarm, never_keeps_a_nan_score_as_the_best)
{
    std::mt19937 rng(6);
    my_torch::SearchResult result = my_torch::ParticleSwarm(PLANE, settings(8, 10)).minimize(nan_on_the_right, rng);

    cr_assert(std::isfinite(result.best_score));
    cr_assert_leq(result.best.at("x"), 0.0);
}

Test(particle_swarm, works_on_whole_numbers)
{
    std::mt19937 rng(7);
    my_torch::SearchSpace sizes({{"hidden", 1, 64, true, true}});
    my_torch::SearchResult result = my_torch::ParticleSwarm(sizes, settings(6, 15)).minimize(distance_to_seven, rng);

    cr_assert_float_eq(result.best.at("hidden"), 7.0, 1e-12);
}

Test(particle_swarm, zero_iterations_only_evaluates_the_starting_positions)
{
    std::mt19937 rng(8);
    my_torch::SearchResult result = my_torch::ParticleSwarm(PLANE, settings(5, 0)).minimize(sphere, rng);

    cr_assert_eq(result.evaluations.size(), 5u);
    cr_assert_eq(result.best_per_iteration.size(), 1u);
}

Test(particle_swarm, rejects_invalid_settings)
{
    my_torch::SwarmSettings no_particles = settings(0, 5);
    my_torch::SwarmSettings negative = settings(5, 5);
    my_torch::SwarmSettings frozen = settings(5, 5);

    negative.inertia = -0.1;
    frozen.max_velocity = 0.0;
    cr_assert_throw(my_torch::ParticleSwarm(PLANE, no_particles), std::invalid_argument);
    cr_assert_throw(my_torch::ParticleSwarm(PLANE, negative), std::invalid_argument);
    cr_assert_throw(my_torch::ParticleSwarm(PLANE, frozen), std::invalid_argument);
}
