/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Activation is the base class: a name (how a saved network stores it, from_name reads it back),
** forward on z, also run as activation(z), and backward from the output gradient to the z one;
** Softmax's derivative is only its Jacobian's diagonal, so it overrides backward with the full one
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "math/matrix.hpp"

namespace my_torch {

class Activation;

using ActivationMaker = std::unique_ptr<Activation> (*)();

class Activation {
public:
    explicit Activation(const std::string &name);
    virtual ~Activation() = default;

    static const std::vector<ActivationMaker> &subclasses();
    static std::vector<std::string> names();
    static std::unique_ptr<Activation> from_name(const std::string &name);

    virtual Matrix forward(const Matrix &z) const = 0;
    virtual Matrix derivative(const Matrix &z) const = 0;
    virtual Matrix backward(const Matrix &z, const Matrix &gradient) const;
    Matrix operator()(const Matrix &z) const;

    std::string name;
};

class Linear : public Activation {
public:
    Linear();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
};

class ReLU : public Activation {
public:
    ReLU();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
};

class Sigmoid : public Activation {
public:
    Sigmoid();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
};

class Tanh : public Activation {
public:
    Tanh();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
};

class Softmax : public Activation {
public:
    Softmax();
    Matrix forward(const Matrix &z) const override;
    Matrix derivative(const Matrix &z) const override;
    Matrix backward(const Matrix &z, const Matrix &gradient) const override;
};

}
