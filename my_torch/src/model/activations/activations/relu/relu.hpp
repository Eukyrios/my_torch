/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** ReLU cuts negative values to 0: its derivative is 1 for positive values, 0 otherwise
*/

#pragma once

#include "model/activations/activation.hpp"

namespace my_torch {

class ReLU : public Activation {
public:
    ReLU();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
};

}
