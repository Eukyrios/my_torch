/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** The hyperparameters to search: each dimension has bounds, a linear or log scale and can be
** whole; positions live in the unit cube [0, 1]^n and decode turns one into real values
*/

#pragma once

#include <cstddef>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace my_torch {

using Hyperparameters = std::map<std::string, double>;

struct Dimension {
    std::string name;
    double low;
    double high;
    bool log_scale = false;
    bool integer = false;
};

class SearchSpace {
public:
    explicit SearchSpace(const std::vector<Dimension> &dimensions);

    std::size_t size() const;
    std::vector<double> random_position(std::mt19937 &rng) const;
    Hyperparameters decode(const std::vector<double> &position) const;

    std::vector<Dimension> dimensions;
};

}
