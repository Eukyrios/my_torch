/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Softmax turns each row of z into probabilities summing to 1, minus the row maximum first so
** exp never overflows; derivative is only its Jacobian's diagonal, so backward uses the full one
*/

#pragma once

#include "model/activations/activation.hpp"

namespace my_torch {

class Softmax : public Activation {
public:
    Softmax();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
    Matrix backward(const Matrix &z, const Matrix &gradient) const override;
};

}
