/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Calling a layer runs its forward pass
*/

#include "layer.hpp"

namespace my_torch {

Matrix Layer::operator()(const Matrix &inputs)
{
    return forward(inputs);
}

}
