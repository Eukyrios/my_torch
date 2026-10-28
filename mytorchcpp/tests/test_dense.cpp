/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the dense layer
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "assert_matrix.hpp"
#include "dense.hpp"
#include "neuron.hpp"

static const my_torch::Matrix WEIGHTS({{0.5, -1.0}, {0.25, 0.5}});
static const std::vector<double> BIASES = {0.1, -0.3};

static std::vector<double> column(const my_torch::Matrix &matrix, std::size_t col)
{
    std::vector<double> values;
    for (std::size_t row = 0; row < matrix.rows(); row++)
        values.push_back(matrix(row, col));
    return values;
}

static std::vector<double> row(const my_torch::Matrix &matrix, std::size_t row)
{
    std::vector<double> values;
    for (std::size_t col = 0; col < matrix.cols(); col++)
        values.push_back(matrix(row, col));
    return values;
}

Test(dense, forward_matches_the_hand_computation)
{
    assert_matrix_eq(my_torch::Dense(WEIGHTS, BIASES).forward(my_torch::Matrix::row({1, 2})), my_torch::Matrix({{1.1, -0.3}}));
}

Test(dense, each_output_is_the_neuron_of_its_column)
{
    my_torch::Matrix weights({{0.3, -0.7, 1.2}, {-0.4, 0.9, 0.1}, {2.0, -1.5, 0.6}, {0.8, 0.2, -0.9}});
    std::vector<double> biases = {0.5, -0.2, 0.05};
    std::vector<double> inputs = {1.5, -0.5, 0.25, 2.0};
    my_torch::Matrix outputs = my_torch::Dense(weights, biases).forward(my_torch::Matrix::row(inputs));

    for (std::size_t j = 0; j < 3; j++)
        cr_assert_float_eq(outputs(0, j), my_torch::weighted_sum(inputs, column(weights, j), biases[j]), 1e-9);
}

Test(dense, sizes_come_from_the_weight_matrix)
{
    my_torch::Dense layer(my_torch::Matrix(768, 256), std::vector<double>(256));

    cr_assert_eq(layer.n_inputs(), 768u);
    cr_assert_eq(layer.n_outputs(), 256u);
}

Test(dense, batch_gives_one_output_row_per_sample)
{
    my_torch::Matrix outputs = my_torch::Dense(my_torch::Matrix(768, 3), std::vector<double>(3)).forward(my_torch::Matrix(32, 768));

    cr_assert_eq(outputs.rows(), 32u);
    cr_assert_eq(outputs.cols(), 3u);
}

Test(dense, each_batch_row_matches_the_single_sample_forward)
{
    my_torch::Dense layer(my_torch::Matrix({{0.3, -0.7}, {-0.4, 0.9}, {2.0, -1.5}}), {0.5, -0.2});
    my_torch::Matrix batch({{1, 2, 3}, {-1, 0.5, 0}, {0.25, -2, 4}});
    my_torch::Matrix outputs = layer.forward(batch);

    for (std::size_t sample = 0; sample < batch.rows(); sample++)
        assert_matrix_eq(my_torch::Matrix::row(row(outputs, sample)), layer.forward(my_torch::Matrix::row(row(batch, sample))));
}

Test(dense, biases_are_added_to_every_row)
{
    my_torch::Matrix outputs = my_torch::Dense(my_torch::Matrix(2, 2), BIASES).forward(my_torch::Matrix(3, 2, 1.0));

    assert_matrix_eq(outputs, my_torch::Matrix({{0.1, -0.3}, {0.1, -0.3}, {0.1, -0.3}}));
}

Test(dense, forward_applies_the_activation)
{
    assert_matrix_eq(my_torch::Dense(WEIGHTS, BIASES, std::make_unique<my_torch::ReLU>()).forward(my_torch::Matrix::row({1, 2})), my_torch::Matrix::row({1.1, 0.0}));
}

Test(dense, default_activation_is_linear)
{
    cr_assert_str_eq(my_torch::Dense(WEIGHTS, BIASES).activation->name().c_str(), "linear");
}

Test(dense, forward_keeps_its_inputs_and_z_for_backpropagation)
{
    my_torch::Dense layer(WEIGHTS, BIASES, std::make_unique<my_torch::ReLU>());

    layer.forward(my_torch::Matrix::row({1, 2}));
    assert_matrix_eq(layer.inputs, my_torch::Matrix::row({1, 2}));
    assert_matrix_eq(layer.z, my_torch::Matrix::row({1.1, -0.3}));
}

Test(dense, softmax_layer_outputs_one_probability_row_per_sample)
{
    my_torch::Dense layer(my_torch::Matrix({{0.3, -0.7, 1.2}, {-0.4, 0.9, 0.1}}), {0.5, -0.2, 0.05}, std::make_unique<my_torch::Softmax>());
    my_torch::Matrix outputs = layer.forward(my_torch::Matrix({{1, 2}, {-1, 0.5}, {3, -2}, {0, 0}}));

    assert_matrix_eq(outputs.row_sums(), my_torch::Matrix({{1}, {1}, {1}, {1}}));
}

Test(dense, rejects_empty_weights)
{
    cr_assert_throw(my_torch::Dense(my_torch::Matrix(), {}), std::invalid_argument);
}

Test(dense, rejects_a_bias_count_that_does_not_match_the_neurons)
{
    cr_assert_throw(my_torch::Dense(WEIGHTS, {0.1}), std::invalid_argument);
    cr_assert_throw(my_torch::Dense(WEIGHTS, {0.1, 0.2, 0.3}), std::invalid_argument);
}

Test(dense, rejects_the_wrong_number_of_inputs)
{
    my_torch::Dense layer(WEIGHTS, BIASES);

    cr_assert_throw(layer.forward(my_torch::Matrix::row({1})), std::invalid_argument);
    cr_assert_throw(layer.forward(my_torch::Matrix::row({1, 2, 3})), std::invalid_argument);
}
