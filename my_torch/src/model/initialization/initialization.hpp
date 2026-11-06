/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** How a dense layer draws its first weights: N(0, scale) with He, Xavier or LeCun's scale, or
** Auto (He before ReLU, Xavier otherwise); each has the name a config file uses for it
*/

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "model/activations/activation.hpp"

namespace my_torch {

enum class Initialization { Auto, He, Xavier, LeCun };

Initialization initialization_from_name(const std::string &name);
std::string initialization_name(Initialization initialization);
std::vector<std::string> initialization_names();
double initial_scale(Initialization initialization, std::size_t n_inputs, std::size_t n_outputs, const Activation &activation);

}
