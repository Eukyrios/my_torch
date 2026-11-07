/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Each step, v = inertia * v + cognitive * r1 * (own best - x) + social * r2 * (swarm best - x),
** capped to max_velocity, then x += v inside [0, 1]; a NaN or infinite score counts as the worst; in
** parallel, scores are recorded in particle order so a run gives the same result every time
*/

#include <algorithm>
#include <exception>
#include <thread>
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

static void record(Swarm &swarm, std::size_t index, std::size_t iteration, const Hyperparameters &hyperparameters, double score, SearchResult &result)
{
    Particle &particle = swarm.particles[index];
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

static void evaluate(Swarm &swarm, std::size_t index, std::size_t iteration, const SearchSpace &space, const Objective &objective, SearchResult &result)
{
    Hyperparameters hyperparameters = space.decode(swarm.particles[index].position);
    record(swarm, index, iteration, hyperparameters, usable_score(objective(hyperparameters)), result);
}

static void score_into(const Objective *objective, const Hyperparameters *hyperparameters, double *score, std::exception_ptr *error)
{
    try {
        *score = usable_score((*objective)(*hyperparameters));
    } catch (...) {
        *error = std::current_exception();
    }
}

static void score_in_parallel(const Objective &objective, const std::vector<Hyperparameters> &values, std::vector<double> &scores, std::size_t threads)
{
    std::vector<std::exception_ptr> errors(values.size());
    for (std::size_t start = 0; start < values.size(); start += threads) {
        std::vector<std::thread> workers;
        for (std::size_t index = start; index < std::min(values.size(), start + threads); index++)
            workers.emplace_back(score_into, &objective, &values[index], &scores[index], &errors[index]);
        for (std::thread &worker : workers)
            worker.join();
    }
    for (const std::exception_ptr &error : errors)
        if (error)
            std::rethrow_exception(error);
}

static void evaluate_all(Swarm &swarm, std::size_t iteration, const SearchSpace &space, const Objective &objective, std::size_t threads, SearchResult &result)
{
    std::vector<Hyperparameters> values;
    for (const Particle &particle : swarm.particles)
        values.push_back(space.decode(particle.position));
    std::vector<double> scores(values.size());
    score_in_parallel(objective, values, scores, threads);
    for (std::size_t index = 0; index < values.size(); index++)
        record(swarm, index, iteration, values[index], scores[index], result);
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

static std::vector<double> start_position(const std::vector<double> &start, std::size_t size)
{
    if (start.size() != size)
        throw std::invalid_argument("a starting position needs one coordinate per hyperparameter (" + std::to_string(size) + "), got " + std::to_string(start.size()));
    std::vector<double> position;
    for (double coordinate : start)
        position.push_back(std::clamp(coordinate, 0.0, 1.0));
    return position;
}

ParticleSwarm::ParticleSwarm(const SearchSpace &space, const SwarmSettings &settings)
:
    space(space), settings(settings)
{
    check_settings(settings);
}

SearchResult ParticleSwarm::minimize(const Objective &objective, std::mt19937 &rng, const std::vector<std::vector<double>> &starts) const
{
    SearchResult result;
    Swarm swarm;
    bool parallel = settings.threads > 1;
    for (std::size_t index = 0; index < settings.particles; index++) {
        std::vector<double> position = space.random_position(rng);
        if (index < starts.size())
            position = start_position(starts[index], space.size());
        swarm.particles.push_back({position, random_velocity(space.size(), settings.max_velocity, rng), {}, 0.0});
        if (!parallel)
            evaluate(swarm, index, 0, space, objective, result);
    }
    if (parallel)
        evaluate_all(swarm, 0, space, objective, settings.threads, result);
    result.best_per_iteration.push_back(swarm.best_score);
    for (std::size_t iteration = 1; iteration <= settings.iterations; iteration++) {
        for (std::size_t index = 0; index < settings.particles; index++) {
            move(swarm.particles[index], swarm.best_position, settings, rng);
            if (!parallel)
                evaluate(swarm, index, iteration, space, objective, result);
        }
        if (parallel)
            evaluate_all(swarm, iteration, space, objective, settings.threads, result);
        result.best_per_iteration.push_back(swarm.best_score);
    }
    result.best = space.decode(swarm.best_position);
    result.best_score = swarm.best_score;
    return result;
}

}
