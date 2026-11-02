/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tanh squashes z into (-1, 1): its derivative is 1 - tanh^2
*/

#pragma once

#include "model/activations/activation.hpp"

namespace my_torch {

class Tanh : public Activation {
public:
    Tanh();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
};

}
