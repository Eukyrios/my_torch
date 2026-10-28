/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A network runs its layers in order on a batch, network(inputs); its sizes come from its
** first and last dense layers, which must chain
*/

#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "dense_layer.hpp"
#include "layer.hpp"

namespace my_torch {

class Network {
public:
    explicit Network(std::vector<std::unique_ptr<Layer>> layers);

    std::vector<DenseLayer *> dense_layers() const;
    std::size_t n_inputs() const;
    std::size_t n_outputs() const;
    Matrix forward(const Matrix &inputs);
    Matrix operator()(const Matrix &inputs);

    std::vector<std::unique_ptr<Layer>> layers;
};

}
