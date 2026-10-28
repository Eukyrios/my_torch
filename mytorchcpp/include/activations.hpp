/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Activation is the base class: each subclass gives its name, how a saved network stores it,
** and its forward pass on z; an activation is called like a function, activation(z), and
** Activation::from_name turns a stored name back into one by trying each of subclasses()
*/

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "matrix.hpp"

namespace my_torch {

class Activation;

using ActivationMaker = std::unique_ptr<Activation> (*)();

class Activation {
public:
    virtual ~Activation() = default;

    static const std::vector<ActivationMaker> &subclasses();
    static std::vector<std::string> names();
    static std::unique_ptr<Activation> from_name(const std::string &name);

    virtual std::string name() const = 0;
    virtual Matrix forward(const Matrix &z) const = 0;
    Matrix operator()(const Matrix &z) const;
};

class Linear : public Activation {
public:
    std::string name() const override;
    Matrix forward(const Matrix &z) const override;
};

class ReLU : public Activation {
public:
    std::string name() const override;
    Matrix forward(const Matrix &z) const override;
};

class Sigmoid : public Activation {
public:
    std::string name() const override;
    Matrix forward(const Matrix &z) const override;
};

class Tanh : public Activation {
public:
    std::string name() const override;
    Matrix forward(const Matrix &z) const override;
};

class Softmax : public Activation {
public:
    std::string name() const override;
    Matrix forward(const Matrix &z) const override;
};

}
