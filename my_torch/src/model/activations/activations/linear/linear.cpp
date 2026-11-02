/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Linear keeps z as it is: its derivative is 1 everywhere
*/

#include "model/activations/activations/linear/linear.hpp"

namespace my_torch {

Linear::Linear()
:
    Activation("linear")
{}

Matrix Linear::forward(const Matrix &z) const
{
    return z;
}

Matrix Linear::derivative(const Matrix &z) const
{
    return Matrix(z.rows(), z.cols(), 1.0);
}

}
