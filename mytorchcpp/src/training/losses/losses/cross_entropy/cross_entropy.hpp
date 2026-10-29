/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** CrossEntropy compares softmax probabilities to one-hot outputs; predictions are clipped to
** at least epsilon (1e-12 by default) so log never sees 0
*/

#pragma once

#include "training/losses/loss.hpp"

namespace my_torch {

class CrossEntropy : public Loss {
public:
    explicit CrossEntropy(double epsilon = 1e-12);

    double forward(const Matrix &predictions, const Matrix &expected) const override;
    Matrix gradient(const Matrix &predictions, const Matrix &expected) const override;

    double epsilon;
};

}
