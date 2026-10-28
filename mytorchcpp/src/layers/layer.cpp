/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Every layer has a name, and calling a layer runs its forward pass
*/

#include "layers/layer.hpp"

namespace my_torch {

Layer::Layer(const std::string &name) : name(name)
{
}

std::optional<Shape> Layer::shape() const
{
    return std::nullopt;
}

std::vector<Parameter> Layer::parameters()
{
    return {};
}

Matrix Layer::operator()(const Matrix &inputs)
{
    return forward(inputs);
}

}
