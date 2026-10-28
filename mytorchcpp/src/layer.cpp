/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Every layer has a name, and calling a layer runs its forward pass
*/

#include "layer.hpp"

namespace my_torch {

Layer::Layer(const std::string &name) : name(name)
{
}

std::optional<std::size_t> Layer::n_inputs() const
{
    return std::nullopt;
}

std::optional<std::size_t> Layer::n_outputs() const
{
    return std::nullopt;
}

Matrix Layer::operator()(const Matrix &inputs)
{
    return forward(inputs);
}

}
