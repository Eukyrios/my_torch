/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Activation functions applied to a layer's weighted sums z. An Activation holds its name,
** how a saved network will store it, and its function, and is called like the function
*/

#pragma once

#include <string>
#include <vector>

#include "matrix.hpp"

namespace my_torch {

using ActivationFunction = Matrix (*)(const Matrix &);

Matrix linear(const Matrix &z);
Matrix relu(const Matrix &z);
Matrix sigmoid(const Matrix &z);
Matrix tanh(const Matrix &z);
Matrix softmax(const Matrix &z);

struct Activation {
    std::string name;
    ActivationFunction function;

    Matrix operator()(const Matrix &z) const;
};

extern const Activation LINEAR;
extern const Activation RELU;
extern const Activation SIGMOID;
extern const Activation TANH;
extern const Activation SOFTMAX;

const std::vector<Activation> &all_activations();
std::vector<std::string> activation_names();
Activation find_activation(const std::string &name);

}
