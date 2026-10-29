/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Linear keeps z as it is: its derivative is 1 everywhere
*/

#pragma once

#include "model/activations/activation.hpp"

namespace my_torch {

class Linear : public Activation {
public:
    Linear();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
};

}
