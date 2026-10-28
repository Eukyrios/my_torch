/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A single neuron: z = x1*w1 + ... + xn*wn + b, the value an activation is applied to
*/

#pragma once

#include <vector>

namespace my_torch {

double weighted_sum(const std::vector<double> &inputs, const std::vector<double> &weights, double bias);

}
