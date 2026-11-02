/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Sigmoid squashes z into (0, 1), computed as 0.5 * (1 + tanh(z / 2)) so exp never overflows;
** its derivative is sigmoid * (1 - sigmoid)
*/

#pragma once

#include "model/activations/activation.hpp"

namespace my_torch {

class Sigmoid : public Activation {
public:
    Sigmoid();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
};

}
