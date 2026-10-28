/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the Layer base class
*/

#include <criterion/criterion.h>

#include "assert_matrix.hpp"
#include "layer.hpp"

class Double : public my_torch::Layer {
public:
    my_torch::Matrix forward(const my_torch::Matrix &inputs) override
    {
        return inputs * 2.0;
    }
};

Test(layer, is_called_like_its_forward)
{
    Double layer;
    my_torch::Matrix inputs({{1, -2}, {0.5, 3}});

    assert_matrix_eq(layer(inputs), layer.forward(inputs));
}

Test(layer, is_used_through_the_base_class)
{
    Double layer;
    my_torch::Layer &base = layer;

    assert_matrix_eq(base(my_torch::Matrix::row({1, 2})), my_torch::Matrix::row({2, 4}));
}
