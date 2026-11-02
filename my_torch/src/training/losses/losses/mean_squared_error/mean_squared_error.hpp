/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** MeanSquaredError is the mean of (prediction - expected)^2 over every value
*/

#pragma once

#include "training/losses/loss.hpp"

namespace my_torch {

class MeanSquaredError : public Loss {
public:
    MeanSquaredError();

    double forward(const Matrix &predictions, const Matrix &expected) const override;
    Matrix gradient(const Matrix &predictions, const Matrix &expected) const override;
};

}
