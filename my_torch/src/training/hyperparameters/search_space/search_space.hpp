/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** The hyperparameters to search: each dimension has bounds, a linear or log scale and can be whole,
** or is a choice between named options; positions live in [0, 1]^n and decode turns one into values
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
    std::vector<std::string> choices = {};
};

Dimension categorical(const std::string &name, const std::vector<std::string> &choices);

class SearchSpace {
public:
    explicit SearchSpace(const std::vector<Dimension> &dimensions);

    std::size_t size() const;
    std::vector<double> random_position(std::mt19937 &rng) const;
    Hyperparameters decode(const std::vector<double> &position) const;

    std::vector<Dimension> dimensions;
};

}
