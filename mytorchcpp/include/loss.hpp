/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Loss is the base class: a name, forward giving how wrong the predictions are averaged over
** the batch (also run by calling it), and gradient, where backpropagation starts
*/

#pragma once

#include <string>

#include "matrix.hpp"

namespace my_torch {

class Loss {
public:
    explicit Loss(const std::string &name);
    virtual ~Loss() = default;

    virtual double forward(const Matrix &predictions, const Matrix &expected) const = 0;
    virtual Matrix gradient(const Matrix &predictions, const Matrix &expected) const = 0;
    double operator()(const Matrix &predictions, const Matrix &expected) const;

    std::string name;
};

class MeanSquaredError : public Loss {
public:
    MeanSquaredError();

    double forward(const Matrix &predictions, const Matrix &expected) const override;
    Matrix gradient(const Matrix &predictions, const Matrix &expected) const override;
};

class CrossEntropy : public Loss {
public:
    explicit CrossEntropy(double epsilon = 1e-12);

    double forward(const Matrix &predictions, const Matrix &expected) const override;
    Matrix gradient(const Matrix &predictions, const Matrix &expected) const override;

    double epsilon;
};

}
