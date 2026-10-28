/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the activation functions
*/

#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "activations/activation.hpp"
#include "assert_matrix.hpp"

static const my_torch::Matrix Z = my_torch::Matrix::row({-2.0, -0.5, 0.0, 0.5, 2.0});

Test(activations, linear_returns_z_unchanged)
{
    assert_matrix_eq(my_torch::Linear()(Z), Z);
}

Test(activations, relu_cuts_negative_values_to_zero)
{
    assert_matrix_eq(my_torch::ReLU()(Z), my_torch::Matrix::row({0.0, 0.0, 0.0, 0.5, 2.0}));
}

Test(activations, sigmoid_matches_its_definition)
{
    my_torch::Matrix outputs = my_torch::Sigmoid()(Z);

    for (std::size_t col = 0; col < Z.cols(); col++)
        cr_assert_float_eq(outputs(0, col), 1.0 / (1.0 + std::exp(-Z(0, col))), 1e-9);
}

Test(activations, sigmoid_is_one_half_at_zero)
{
    cr_assert_float_eq(my_torch::Sigmoid()(my_torch::Matrix::row({0.0}))(0, 0), 0.5, 1e-12);
}

Test(activations, sigmoid_does_not_overflow_on_large_values)
{
    assert_matrix_eq(my_torch::Sigmoid()(my_torch::Matrix::row({-1000.0, 1000.0})), my_torch::Matrix::row({0.0, 1.0}));
}

Test(activations, tanh_stays_between_minus_one_and_one)
{
    assert_matrix_eq(my_torch::Tanh()(my_torch::Matrix::row({-1000.0, 0.0, 1000.0})), my_torch::Matrix::row({-1.0, 0.0, 1.0}));
}

Test(activations, softmax_matches_the_hand_computation)
{
    my_torch::Matrix outputs = my_torch::Softmax()(my_torch::Matrix::row({2.0, 1.0, 0.1}));

    cr_assert_float_eq(outputs(0, 0), 0.659, 1e-3);
    cr_assert_float_eq(outputs(0, 1), 0.242, 1e-3);
    cr_assert_float_eq(outputs(0, 2), 0.099, 1e-3);
}

Test(activations, softmax_rows_sum_to_one)
{
    my_torch::Matrix batch({{0.3, -1.2, 2.5}, {4.0, 4.0, -3.0}, {-0.7, 0.0, 0.1}});

    assert_matrix_eq(my_torch::Softmax()(batch).row_sums(), my_torch::Matrix({{1}, {1}, {1}}));
}

Test(activations, softmax_works_row_by_row_on_a_batch)
{
    my_torch::Matrix outputs = my_torch::Softmax()(my_torch::Matrix({{2.0, 1.0, 0.1}, {0.0, 0.0, 0.0}}));
    my_torch::Matrix first = my_torch::Softmax()(my_torch::Matrix::row({2.0, 1.0, 0.1}));

    for (std::size_t col = 0; col < 3; col++) {
        cr_assert_float_eq(outputs(0, col), first(0, col), 1e-12);
        cr_assert_float_eq(outputs(1, col), 1.0 / 3.0, 1e-12);
    }
}

Test(activations, softmax_does_not_overflow_on_large_scores)
{
    assert_matrix_eq(my_torch::Softmax()(my_torch::Matrix::row({1000.0, 1000.0, 1000.0})), my_torch::Matrix::row({1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}));
}

static my_torch::Matrix numerical_derivative(const my_torch::Activation &activation, const my_torch::Matrix &z)
{
    const double step = 1e-6;
    return (activation(z + my_torch::Matrix(z.rows(), z.cols(), step)) - activation(z - my_torch::Matrix(z.rows(), z.cols(), step))) / (2 * step);
}

static void assert_close(const my_torch::Matrix &actual, const my_torch::Matrix &expected, double tolerance)
{
    for (std::size_t row = 0; row < expected.rows(); row++)
        for (std::size_t col = 0; col < expected.cols(); col++)
            cr_assert_float_eq(actual(row, col), expected(row, col), tolerance);
}

Test(activations, linear_derivative_is_one)
{
    assert_matrix_eq(my_torch::Linear().derivative(Z), my_torch::Matrix(1, 5, 1.0));
}

Test(activations, relu_derivative_is_one_for_positive_values_only)
{
    assert_matrix_eq(my_torch::ReLU().derivative(Z), my_torch::Matrix::row({0, 0, 0, 1, 1}));
}

Test(activations, sigmoid_derivative_matches_the_numerical_derivative)
{
    assert_close(my_torch::Sigmoid().derivative(Z), numerical_derivative(my_torch::Sigmoid(), Z), 1e-6);
}

Test(activations, tanh_derivative_matches_the_numerical_derivative)
{
    assert_close(my_torch::Tanh().derivative(Z), numerical_derivative(my_torch::Tanh(), Z), 1e-6);
}

Test(activations, softmax_derivative_is_the_diagonal_of_its_jacobian)
{
    my_torch::Matrix z = my_torch::Matrix::row({2.0, 1.0, 0.1});
    my_torch::Matrix softmax = my_torch::Softmax()(z);
    my_torch::Matrix derivative = my_torch::Softmax().derivative(z);

    for (std::size_t col = 0; col < 3; col++)
        cr_assert_float_eq(derivative(0, col), softmax(0, col) * (1 - softmax(0, col)), 1e-12);
}

Test(activations, an_activation_is_called_like_its_forward)
{
    my_torch::ReLU relu;

    assert_matrix_eq(relu(Z), relu.forward(Z));
}

Test(activations, from_name_finds_every_activation_by_name)
{
    for (const std::string &name : my_torch::Activation::names())
        cr_assert(my_torch::Activation::from_name(name)->name == name);
    cr_assert_not_null(dynamic_cast<my_torch::ReLU *>(my_torch::Activation::from_name("relu").get()));
}

Test(activations, subclasses_make_one_of_each_activation)
{
    cr_assert_eq(my_torch::Activation::subclasses().size(), 5u);
}

Test(activations, names_list_every_activation)
{
    std::vector<std::string> expected = {"linear", "relu", "sigmoid", "tanh", "softmax"};

    cr_assert(my_torch::Activation::names() == expected);
}

Test(activations, from_name_rejects_an_unknown_name)
{
    cr_assert_throw(my_torch::Activation::from_name("swish"), std::invalid_argument);
}
