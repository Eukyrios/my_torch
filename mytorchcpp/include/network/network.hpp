/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A network runs its layers in order on a batch, network(inputs); its sizes come from its
** first and last layers with a fixed size, which must chain
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
    Matrix forward(const Matrix &inputs);
    Matrix operator()(const Matrix &inputs);

    std::vector<std::unique_ptr<Layer>> layers;
};

}
