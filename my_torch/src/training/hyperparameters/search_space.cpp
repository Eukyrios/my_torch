/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A coordinate u in [0, 1] becomes low + u * (high - low), or the same between log(low) and
** log(high) on a log scale, so a learning rate is searched as evenly in 1e-4..1e-3 as in 0.01..0.1
*/

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

#include "training/hyperparameters/search_space.hpp"

namespace my_torch {

static void check_dimension(const Dimension &dimension)
{
    if (dimension.name.empty())
        throw std::invalid_argument("a hyperparameter needs a name");
    if (!(dimension.low < dimension.high))
        throw std::invalid_argument("hyperparameter '" + dimension.name + "' needs low < high");
    if (dimension.log_scale && dimension.low <= 0.0)
        throw std::invalid_argument("hyperparameter '" + dimension.name + "' needs positive bounds on a log scale");
}

static void check_dimensions(const std::vector<Dimension> &dimensions)
{
    if (dimensions.empty())
        throw std::invalid_argument("a search space needs at least one hyperparameter");
    std::set<std::string> names;
    for (const Dimension &dimension : dimensions) {
        check_dimension(dimension);
        if (!names.insert(dimension.name).second)
            throw std::invalid_argument("hyperparameter '" + dimension.name + "' is given twice");
    }
}

static void check_position(const std::vector<double> &position, std::size_t size)
{
    if (position.size() != size)
        throw std::invalid_argument("a position needs one coordinate per hyperparameter (" + std::to_string(size) + "), got " + std::to_string(position.size()));
}

static double decode_coordinate(const Dimension &dimension, double coordinate)
{
    double unit = std::clamp(coordinate, 0.0, 1.0);
    double value = dimension.log_scale ? std::exp(std::log(dimension.low) + unit * (std::log(dimension.high) - std::log(dimension.low))) : dimension.low + unit * (dimension.high - dimension.low);
    if (dimension.integer)
        value = std::round(value);
    return std::clamp(value, dimension.low, dimension.high);
}

SearchSpace::SearchSpace(const std::vector<Dimension> &dimensions)
:
    dimensions(dimensions)
{
    check_dimensions(dimensions);
}

std::size_t SearchSpace::size() const
{
    return dimensions.size();
}

std::vector<double> SearchSpace::random_position(std::mt19937 &rng) const
{
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<double> position;
    for (std::size_t i = 0; i < dimensions.size(); i++)
        position.push_back(unit(rng));
    return position;
}

Hyperparameters SearchSpace::decode(const std::vector<double> &position) const
{
    check_position(position, dimensions.size());
    Hyperparameters values;
    for (std::size_t i = 0; i < dimensions.size(); i++)
        values[dimensions[i].name] = decode_coordinate(dimensions[i], position[i]);
    return values;
}

}
