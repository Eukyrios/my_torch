/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Each step, v = inertia * v + cognitive * r1 * (own best - x) + social * r2 * (swarm best - x),
** capped to max_velocity, then x += v inside [0, 1]; a NaN or infinite score counts as the worst
*/

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "training/hyperparameters/algorithms/particle_swarm/particle_swarm.hpp"

namespace my_torch {

struct Particle {
    std::vector<double> position;
    std::vector<double> velocity;
    std::vector<double> best_position;
    double best_score;
};

struct Swarm {
    std::vector<Particle> particles;
    std::vector<double> best_position;
    double best_score = std::numeric_limits<double>::infinity();
};

static void check_settings(const SwarmSettings &settings)
{
    if (settings.particles < 1)
        throw std::invalid_argument("a swarm needs at least 1 particle");
    if (settings.inertia < 0.0 || settings.cognitive < 0.0 || settings.social < 0.0)
        throw std::invalid_argument("the inertia, cognitive and social weights must not be negative");
    if (!(settings.max_velocity > 0.0))
        throw std::invalid_argument("the maximum velocity must be positive, got " + std::to_string(settings.max_velocity));
}

static double usable_score(double score)
{
    return std::isfinite(score) ? score : std::numeric_limits<double>::infinity();
}

static std::vector<double> random_velocity(std::size_t size, double max_velocity, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> speed(-max_velocity, max_velocity);
    std::vector<double> velocity;
    for (std::size_t i = 0; i < size; i++)
        velocity.push_back(speed(rng));
    return velocity;
}

static void evaluate(Swarm &swarm, std::size_t index, std::size_t iteration, const SearchSpace &space, const Objective &objective, SearchResult &result)
{
    Particle &particle = swarm.particles[index];
    Hyperparameters hyperparameters = space.decode(particle.position);
    double score = usable_score(objective(hyperparameters));
    result.evaluations.push_back({iteration, index, hyperparameters, score});
    if (particle.best_position.empty() || score < particle.best_score) {
        particle.best_position = particle.position;
        particle.best_score = score;
    }
    if (swarm.best_position.empty() || score < swarm.best_score) {
        swarm.best_position = particle.position;
        swarm.best_score = score;
    }
}

static void move(Particle &particle, const std::vector<double> &swarm_best, const SwarmSettings &settings, std::mt19937 &rng)
{
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::size_t d = 0; d < particle.position.size(); d++) {
        double towards_own = settings.cognitive * unit(rng) * (particle.best_position[d] - particle.position[d]);
        double towards_swarm = settings.social * unit(rng) * (swarm_best[d] - particle.position[d]);
        double velocity = settings.inertia * particle.velocity[d] + towards_own + towards_swarm;
        particle.velocity[d] = std::clamp(velocity, -settings.max_velocity, settings.max_velocity);
        particle.position[d] += particle.velocity[d];
        if (particle.position[d] < 0.0 || particle.position[d] > 1.0) {
            particle.position[d] = std::clamp(particle.position[d], 0.0, 1.0);
            particle.velocity[d] = 0.0;
        }
    }
}

ParticleSwarm::ParticleSwarm(const SearchSpace &space, const SwarmSettings &settings)
:
    space(space), settings(settings)
{
    check_settings(settings);
}

SearchResult ParticleSwarm::minimize(const Objective &objective, std::mt19937 &rng) const
{
    SearchResult result;
    Swarm swarm;
    for (std::size_t index = 0; index < settings.particles; index++) {
        swarm.particles.push_back({space.random_position(rng), random_velocity(space.size(), settings.max_velocity, rng), {}, 0.0});
        evaluate(swarm, index, 0, space, objective, result);
    }
    result.best_per_iteration.push_back(swarm.best_score);
    for (std::size_t iteration = 1; iteration <= settings.iterations; iteration++) {
        for (std::size_t index = 0; index < settings.particles; index++) {
            move(swarm.particles[index], swarm.best_position, settings, rng);
            evaluate(swarm, index, iteration, space, objective, result);
        }
        result.best_per_iteration.push_back(swarm.best_score);
    }
    result.best = space.decode(swarm.best_position);
    result.best_score = swarm.best_score;
    return result;
}

}
