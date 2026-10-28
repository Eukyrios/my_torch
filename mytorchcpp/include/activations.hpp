/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Activation functions applied to a layer's weighted sums z, found by the name a saved
** network will store them under
*/

#pragma once

#include <map>
#include <string>

#include "matrix.hpp"

namespace my_torch {

using ActivationFunction = Matrix (*)(const Matrix &);

Matrix linear(const Matrix &z);
Matrix relu(const Matrix &z);
Matrix sigmoid(const Matrix &z);
Matrix tanh(const Matrix &z);

const std::map<std::string, ActivationFunction> &activations();
ActivationFunction get_activation(const std::string &name);

}
