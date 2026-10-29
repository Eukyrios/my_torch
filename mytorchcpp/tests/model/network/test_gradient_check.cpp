/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Checks every gradient of a whole network against numerical gradients, (f(x+h) - f(x-h)) / 2h
*/

#include <cmath>
#include <criterion/criterion.h>
#include <random>
#include <string>

#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "training/losses/loss.hpp"
#include "training/losses/losses/mean_squared_error/mean_squared_error.hpp"
#include "model/network/network.hpp"

static my_torch::Matrix random_matrix(std::size_t rows, std::size_t cols, std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    my_torch::Matrix matrix(rows, cols);
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t col = 0; col < cols; col++)
            matrix(row, col) = normal(rng);
    return matrix;
}

static my_torch::Matrix random_labels(std::size_t samples, std::size_t classes, std::mt19937 &rng)
{
    std::uniform_int_distribution<std::size_t> pick(0, classes - 1);
    my_torch::Matrix labels(samples, classes);
    for (std::size_t sample = 0; sample < samples; sample++)
        labels(sample, pick(rng)) = 1.0;
    return labels;
}

static std::vector<double> random_vector(std::size_t size, std::mt19937 &rng)
{
    my_torch::Matrix values = random_matrix(1, size, rng);
    std::vector<double> vector;
    for (std::size_t col = 0; col < size; col++)
        vector.push_back(values(0, col));
    return vector;
}

static std::unique_ptr<my_torch::Layer> random_layer(std::size_t n_inputs, std::size_t n_outputs, const std::string &activation, std::mt19937 &rng)
{
    my_torch::Matrix weights = random_matrix(n_inputs, n_outputs, rng);
    return std::make_unique<my_torch::DenseLayer>(weights, random_vector(n_outputs, rng), my_torch::Activation::from_name(activation));
}

static my_torch::Network build_network(const std::string &hidden, const std::string &output)
{
    std::mt19937 rng(21);
    std::vector<std::unique_ptr<my_torch::Layer>> layers;
    layers.push_back(random_layer(5, 4, hidden, rng));
    layers.push_back(random_layer(4, 4, hidden, rng));
    layers.push_back(random_layer(4, 3, output, rng));
    return my_torch::Network(std::move(layers));
}

struct Batch {
    my_torch::Matrix inputs;
    my_torch::Matrix labels;
};

static Batch random_batch()
{
    std::mt19937 rng(20);
    my_torch::Matrix inputs = random_matrix(6, 5, rng);
    return Batch{inputs, random_labels(6, 3, rng)};
}

static my_torch::Matrix numerical_gradient(my_torch::Network &network, const my_torch::Loss &loss, my_torch::Matrix &inputs, const my_torch::Matrix &labels, my_torch::Matrix &values)
{
    const double step = 1e-6;
    my_torch::Matrix result(values.rows(), values.cols());
    for (std::size_t row = 0; row < values.rows(); row++)
        for (std::size_t col = 0; col < values.cols(); col++) {
            double original = values(row, col);
            values(row, col) = original + step;
            double higher = loss(network(inputs), labels);
            values(row, col) = original - step;
            double lower = loss(network(inputs), labels);
            values(row, col) = original;
            result(row, col) = (higher - lower) / (2 * step);
        }
    return result;
}

static my_torch::Matrix backpropagate(my_torch::Network &network, const my_torch::Loss &loss, const Batch &batch)
{
    my_torch::Matrix predictions = network(batch.inputs);
    return network.backward(loss.gradient(predictions, batch.labels));
}

static void assert_gradient_close(const my_torch::Matrix &computed, const my_torch::Matrix &numerical)
{
    for (std::size_t row = 0; row < numerical.rows(); row++)
        for (std::size_t col = 0; col < numerical.cols(); col++)
            cr_assert_leq(std::abs(computed(row, col) - numerical(row, col)), 1e-7 + 1e-4 * std::abs(numerical(row, col)));
}

static void check_parameter_gradients(const std::string &hidden, const std::string &output, const my_torch::Loss &loss)
{
    my_torch::Network network = build_network(hidden, output);
    Batch batch = random_batch();
    backpropagate(network, loss, batch);
    for (const std::unique_ptr<my_torch::Layer> &layer : network.layers) {
        my_torch::DenseLayer &dense = dynamic_cast<my_torch::DenseLayer &>(*layer);
        my_torch::Matrix weights_gradient = dense.weights_gradient;
        my_torch::Matrix biases_gradient = dense.biases_gradient;
        assert_gradient_close(weights_gradient, numerical_gradient(network, loss, batch.inputs, batch.labels, dense.weights));
        assert_gradient_close(biases_gradient, numerical_gradient(network, loss, batch.inputs, batch.labels, dense.biases));
    }
}

static void check_input_gradient(const std::string &hidden, const std::string &output, const my_torch::Loss &loss)
{
    my_torch::Network network = build_network(hidden, output);
    Batch batch = random_batch();
    my_torch::Matrix input_gradient = backpropagate(network, loss, batch);
    assert_gradient_close(input_gradient, numerical_gradient(network, loss, batch.inputs, batch.labels, batch.inputs));
}

Test(gradient_check, tanh_softmax_cross_entropy_parameters)
{
    check_parameter_gradients("tanh", "softmax", my_torch::CrossEntropy());
}

Test(gradient_check, relu_softmax_cross_entropy_parameters)
{
    check_parameter_gradients("relu", "softmax", my_torch::CrossEntropy());
}

Test(gradient_check, sigmoid_sigmoid_mean_squared_error_parameters)
{
    check_parameter_gradients("sigmoid", "sigmoid", my_torch::MeanSquaredError());
}

Test(gradient_check, tanh_softmax_cross_entropy_inputs)
{
    check_input_gradient("tanh", "softmax", my_torch::CrossEntropy());
}

Test(gradient_check, relu_softmax_cross_entropy_inputs)
{
    check_input_gradient("relu", "softmax", my_torch::CrossEntropy());
}

Test(gradient_check, sigmoid_sigmoid_mean_squared_error_inputs)
{
    check_input_gradient("sigmoid", "sigmoid", my_torch::MeanSquaredError());
}
