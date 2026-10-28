/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the dense layer
*/

#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "assert_matrix.hpp"
#include "layers/dense_layer.hpp"
#include "layers/neuron.hpp"

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

Test(dense_layer, forward_matches_the_hand_computation)
{
    assert_matrix_eq(my_torch::DenseLayer(WEIGHTS, BIASES).forward(my_torch::Matrix::row({1, 2})), my_torch::Matrix({{1.1, -0.3}}));
}

Test(dense_layer, each_output_is_the_neuron_of_its_column)
{
    my_torch::Matrix weights({{0.3, -0.7, 1.2}, {-0.4, 0.9, 0.1}, {2.0, -1.5, 0.6}, {0.8, 0.2, -0.9}});
    std::vector<double> biases = {0.5, -0.2, 0.05};
    std::vector<double> inputs = {1.5, -0.5, 0.25, 2.0};
    my_torch::Matrix outputs = my_torch::DenseLayer(weights, biases).forward(my_torch::Matrix::row(inputs));

    for (std::size_t j = 0; j < 3; j++)
        cr_assert_float_eq(outputs(0, j), my_torch::Neuron(column(weights, j), biases[j]).forward(inputs), 1e-9);
}

Test(dense_layer, shape_comes_from_the_weight_matrix)
{
    my_torch::DenseLayer layer(my_torch::Matrix(768, 256), std::vector<double>(256));

    cr_assert(layer.shape() == my_torch::Shape(768, 256));
}

Test(dense_layer, batch_gives_one_output_row_per_sample)
{
    my_torch::Matrix outputs = my_torch::DenseLayer(my_torch::Matrix(768, 3), std::vector<double>(3)).forward(my_torch::Matrix(32, 768));

    cr_assert_eq(outputs.rows(), 32u);
    cr_assert_eq(outputs.cols(), 3u);
}

Test(dense_layer, each_batch_row_matches_the_single_sample_forward)
{
    my_torch::DenseLayer layer(my_torch::Matrix({{0.3, -0.7}, {-0.4, 0.9}, {2.0, -1.5}}), {0.5, -0.2});
    my_torch::Matrix batch({{1, 2, 3}, {-1, 0.5, 0}, {0.25, -2, 4}});
    my_torch::Matrix outputs = layer.forward(batch);

    for (std::size_t sample = 0; sample < batch.rows(); sample++)
        assert_matrix_eq(my_torch::Matrix::row(row(outputs, sample)), layer.forward(my_torch::Matrix::row(row(batch, sample))));
}

Test(dense_layer, biases_are_added_to_every_row)
{
    my_torch::Matrix outputs = my_torch::DenseLayer(my_torch::Matrix(2, 2), BIASES).forward(my_torch::Matrix(3, 2, 1.0));

    assert_matrix_eq(outputs, my_torch::Matrix({{0.1, -0.3}, {0.1, -0.3}, {0.1, -0.3}}));
}

Test(dense_layer, is_named_dense)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);

    cr_assert_str_eq(layer.name.c_str(), "dense");
}

Test(dense_layer, is_called_like_a_function)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);

    assert_matrix_eq(layer(my_torch::Matrix::row({1, 2})), my_torch::Matrix::row({1.1, -0.3}));
}

Test(dense_layer, forward_applies_the_activation)
{
    assert_matrix_eq(my_torch::DenseLayer(WEIGHTS, BIASES, std::make_unique<my_torch::ReLU>()).forward(my_torch::Matrix::row({1, 2})), my_torch::Matrix::row({1.1, 0.0}));
}

Test(dense_layer, default_activation_is_linear)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);

    cr_assert_str_eq(layer.activation->name.c_str(), "linear");
}

Test(dense_layer, forward_keeps_its_inputs_and_z_for_backpropagation)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES, std::make_unique<my_torch::ReLU>());

    layer.forward(my_torch::Matrix::row({1, 2}));
    assert_matrix_eq(layer.inputs, my_torch::Matrix::row({1, 2}));
    assert_matrix_eq(layer.z, my_torch::Matrix::row({1.1, -0.3}));
}

Test(dense_layer, softmax_layer_outputs_one_probability_row_per_sample)
{
    my_torch::DenseLayer layer(my_torch::Matrix({{0.3, -0.7, 1.2}, {-0.4, 0.9, 0.1}}), {0.5, -0.2, 0.05}, std::make_unique<my_torch::Softmax>());
    my_torch::Matrix outputs = layer.forward(my_torch::Matrix({{1, 2}, {-1, 0.5}, {3, -2}, {0, 0}}));

    assert_matrix_eq(outputs.row_sums(), my_torch::Matrix({{1}, {1}, {1}, {1}}));
}

Test(dense_layer, rejects_empty_weights)
{
    cr_assert_throw(my_torch::DenseLayer(my_torch::Matrix(), {}), std::invalid_argument);
}

Test(dense_layer, rejects_a_bias_count_that_does_not_match_the_neurons)
{
    cr_assert_throw(my_torch::DenseLayer(WEIGHTS, {0.1}), std::invalid_argument);
    cr_assert_throw(my_torch::DenseLayer(WEIGHTS, {0.1, 0.2, 0.3}), std::invalid_argument);
}

Test(dense_layer, rejects_the_wrong_number_of_inputs)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);

    cr_assert_throw(layer.forward(my_torch::Matrix::row({1})), std::invalid_argument);
    cr_assert_throw(layer.forward(my_torch::Matrix::row({1, 2, 3})), std::invalid_argument);
}

static my_torch::DenseLayer layer_after_forward()
{
    my_torch::DenseLayer layer(my_torch::Matrix({{0.5, -1.0, 0.2}, {0.25, 0.5, -0.4}}), {0.1, -0.3, 0.05}, std::make_unique<my_torch::Tanh>());
    layer.forward(my_torch::Matrix({{1, 2}, {-0.5, 0.3}}));
    return layer;
}

Test(dense_layer, backward_with_a_linear_activation_matches_the_hand_computation)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);
    my_torch::Matrix gradient = my_torch::Matrix::row({1.0, -2.0});

    layer.forward(my_torch::Matrix::row({1, 2}));
    assert_matrix_eq(layer.backward(gradient), my_torch::Matrix::row({2.5, -0.75}));
    assert_matrix_eq(layer.weights_gradient, my_torch::Matrix({{1, -2}, {2, -4}}));
    assert_matrix_eq(layer.biases_gradient, gradient);
}

Test(dense_layer, backward_gives_gradients_of_the_parameter_shapes)
{
    my_torch::DenseLayer layer = layer_after_forward();
    my_torch::Matrix input_gradient = layer.backward(my_torch::Matrix(2, 3, 1.0));

    cr_assert_eq(input_gradient.rows(), 2u);
    cr_assert_eq(input_gradient.cols(), 2u);
    cr_assert_eq(layer.weights_gradient.rows(), 2u);
    cr_assert_eq(layer.weights_gradient.cols(), 3u);
    cr_assert_eq(layer.biases_gradient.rows(), 1u);
    cr_assert_eq(layer.biases_gradient.cols(), 3u);
}

Test(dense_layer, backward_sums_the_bias_gradient_over_the_batch)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);

    layer.forward(my_torch::Matrix({{1, 2}, {3, 4}}));
    layer.backward(my_torch::Matrix({{1, 2}, {3, 4}}));
    assert_matrix_eq(layer.biases_gradient, my_torch::Matrix::row({4, 6}));
}

Test(dense_layer, backward_needs_a_forward_first)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);

    cr_assert_throw(layer.backward(my_torch::Matrix::row({1, 2})), std::runtime_error);
}

Test(dense_layer, backward_rejects_a_gradient_of_the_wrong_shape)
{
    my_torch::DenseLayer layer = layer_after_forward();

    cr_assert_throw(layer.backward(my_torch::Matrix(2, 2)), std::invalid_argument);
}

static double standard_deviation(const my_torch::Matrix &values)
{
    double mean = values.sum() / values.size();
    double total = 0.0;
    for (std::size_t row = 0; row < values.rows(); row++)
        for (std::size_t col = 0; col < values.cols(); col++)
            total += (values(row, col) - mean) * (values(row, col) - mean);
    return std::sqrt(total / values.size());
}

Test(dense_layer, create_has_the_requested_shape_and_zero_biases)
{
    std::mt19937 rng(0);
    std::unique_ptr<my_torch::DenseLayer> layer = my_torch::DenseLayer::create(4, 3, nullptr, rng);

    cr_assert(layer->shape() == my_torch::Shape(4, 3));
    assert_matrix_eq(layer->biases, my_torch::Matrix(1, 3));
}

Test(dense_layer, create_uses_he_scale_for_relu)
{
    std::mt19937 rng(0);
    std::unique_ptr<my_torch::DenseLayer> layer = my_torch::DenseLayer::create(400, 300, std::make_unique<my_torch::ReLU>(), rng);

    cr_assert_float_eq(standard_deviation(layer->weights), std::sqrt(2.0 / 400), 0.002);
}

Test(dense_layer, create_uses_xavier_scale_otherwise)
{
    std::mt19937 rng(0);
    std::unique_ptr<my_torch::DenseLayer> layer = my_torch::DenseLayer::create(400, 300, std::make_unique<my_torch::Tanh>(), rng);

    cr_assert_float_eq(standard_deviation(layer->weights), std::sqrt(2.0 / 700), 0.002);
}

Test(dense_layer, create_is_reproducible_from_the_same_seed)
{
    std::mt19937 first(7);
    std::mt19937 second(7);

    assert_matrix_eq(my_torch::DenseLayer::create(3, 2, nullptr, first)->weights, my_torch::DenseLayer::create(3, 2, nullptr, second)->weights);
}

Test(dense_layer, create_keeps_its_activation)
{
    std::mt19937 rng(0);
    std::unique_ptr<my_torch::DenseLayer> layer = my_torch::DenseLayer::create(3, 2, std::make_unique<my_torch::Softmax>(), rng);

    cr_assert(layer->activation->name == "softmax");
}

Test(dense_layer, create_rejects_empty_sizes)
{
    std::mt19937 rng(0);

    cr_assert_throw(my_torch::DenseLayer::create(0, 3, nullptr, rng), std::invalid_argument);
    cr_assert_throw(my_torch::DenseLayer::create(3, 0, nullptr, rng), std::invalid_argument);
}

Test(dense_layer, parameters_are_the_weights_and_biases_with_their_gradients)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);
    std::vector<my_torch::Parameter> parameters = layer.parameters();

    cr_assert_eq(parameters.size(), 2u);
    cr_assert_eq(parameters[0].value, &layer.weights);
    cr_assert_eq(parameters[0].gradient, &layer.weights_gradient);
    cr_assert_eq(parameters[1].value, &layer.biases);
    cr_assert_eq(parameters[1].gradient, &layer.biases_gradient);
}

Test(dense_layer, parameter_gradients_are_empty_before_backward)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);

    cr_assert_eq(layer.parameters()[0].gradient->size(), 0u);
}

Test(dense_layer, parameters_see_the_gradients_after_backward)
{
    my_torch::DenseLayer layer(WEIGHTS, BIASES);

    layer.forward(my_torch::Matrix::row({1, 2}));
    layer.backward(my_torch::Matrix::row({1.0, -2.0}));
    assert_matrix_eq(*layer.parameters()[0].gradient, my_torch::Matrix({{1, -2}, {2, -4}}));
}
