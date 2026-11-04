/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Dropout fights overfitting: while the network trains, each input is set to 0 with probability
** rate and the others are scaled by 1 / (1 - rate), so no neuron can rely on a few others; out
** of training it passes its inputs as they are. It works on any size, so it has no shape
*/

#pragma once

#include <random>

#include "model/layers/layer.hpp"

namespace my_torch {

class Dropout : public Layer {
public:
    explicit Dropout(double rate, std::mt19937 rng = std::mt19937());

    Matrix forward(const Matrix &inputs) override;
    Matrix backward(const Matrix &gradient) override;

    double rate;
    std::mt19937 rng;
    Matrix mask;
};

}
