/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A network runs its layers in order on a batch, network(inputs), and backward runs them in
** reverse, from the loss gradient back to the inputs; its shape comes from its sized layers
*/

#pragma once

#include <memory>
#include <vector>

#include "layers/layer.hpp"

namespace my_torch {

class Network {
public:
    explicit Network(std::vector<std::unique_ptr<Layer>> layers);

    Shape shape() const;
    std::vector<Parameter> parameters();
    Matrix forward(const Matrix &inputs);
    Matrix operator()(const Matrix &inputs);
    Matrix backward(const Matrix &gradient);

    std::vector<std::unique_ptr<Layer>> layers;
};

}
