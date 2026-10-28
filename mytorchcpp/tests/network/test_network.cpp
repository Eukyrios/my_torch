/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the network
*/

#include <criterion/criterion.h>
#include <random>
#include <stdexcept>

#include "assert_matrix.hpp"
#include "layers/dense_layer.hpp"
#include "network/network.hpp"

using LayerList = std::vector<std::unique_ptr<my_torch::Layer>>;

class PassThrough : public my_torch::Layer {
public:
    PassThrough() : Layer("pass_through")
    {
    }

    my_torch::Matrix forward(const my_torch::Matrix &inputs) override
    {
        return inputs;
    }
};

static std::unique_ptr<my_torch::Layer> hidden_layer()
{
    return std::make_unique<my_torch::DenseLayer>(my_torch::Matrix({{0.5, -1.0}, {0.25, 0.5}}), std::vector<double>{0.1, -0.3}, std::make_unique<my_torch::ReLU>());
}

static std::unique_ptr<my_torch::Layer> output_layer()
{
    return std::make_unique<my_torch::DenseLayer>(my_torch::Matrix({{1.0, 0.5, -1.0}, {0.0, 1.0, 1.0}}), std::vector<double>{0.0, 0.0, 0.0}, std::make_unique<my_torch::Softmax>());
}

static LayerList layers_of(std::unique_ptr<my_torch::Layer> first, std::unique_ptr<my_torch::Layer> second)
{
    LayerList layers;
    layers.push_back(std::move(first));
    layers.push_back(std::move(second));
    return layers;
}

static my_torch::Matrix random_matrix(std::size_t rows, std::size_t cols, std::mt19937 &rng)
{
    std::normal_distribution<double> normal(0.0, 1.0);
    my_torch::Matrix matrix(rows, cols);
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t col = 0; col < cols; col++)
            matrix(row, col) = normal(rng);
    return matrix;
}

static my_torch::Network random_network(const std::vector<std::size_t> &sizes)
{
    std::mt19937 rng(0);
    LayerList layers;
    for (std::size_t i = 0; i + 1 < sizes.size(); i++)
        layers.push_back(std::make_unique<my_torch::DenseLayer>(random_matrix(sizes[i], sizes[i + 1], rng), std::vector<double>(sizes[i + 1], 0.1), std::make_unique<my_torch::ReLU>()));
    return my_torch::Network(std::move(layers));
}

static my_torch::Matrix row_of(const my_torch::Matrix &matrix, std::size_t row)
{
    my_torch::Matrix values(1, matrix.cols());
    for (std::size_t col = 0; col < matrix.cols(); col++)
        values(0, col) = matrix(row, col);
    return values;
}

Test(network, forward_matches_the_hand_computation)
{
    my_torch::Network network(layers_of(hidden_layer(), output_layer()));
    my_torch::Matrix outputs = network.forward(my_torch::Matrix::row({1, 2}));

    cr_assert_float_eq(outputs(0, 0), 0.593, 1e-3);
    cr_assert_float_eq(outputs(0, 1), 0.342, 1e-3);
    cr_assert_float_eq(outputs(0, 2), 0.066, 1e-3);
}

Test(network, forward_runs_the_layers_in_order)
{
    std::unique_ptr<my_torch::Layer> hidden = hidden_layer();
    std::unique_ptr<my_torch::Layer> output = output_layer();
    my_torch::Matrix inputs = my_torch::Matrix::row({1, 2});
    my_torch::Matrix expected = (*output)((*hidden)(inputs));

    assert_matrix_eq(my_torch::Network(layers_of(std::move(hidden), std::move(output))).forward(inputs), expected);
}

Test(network, calling_a_network_runs_its_forward)
{
    my_torch::Network network(layers_of(hidden_layer(), output_layer()));
    my_torch::Matrix inputs = my_torch::Matrix::row({1, 2});

    assert_matrix_eq(network(inputs), network.forward(inputs));
}

Test(network, a_batch_gives_one_output_row_per_sample)
{
    my_torch::Network network = random_network({768, 64, 3});
    my_torch::Matrix outputs = network(my_torch::Matrix(32, 768));

    cr_assert_eq(outputs.rows(), 32u);
    cr_assert_eq(outputs.cols(), 3u);
}

Test(network, each_batch_row_matches_the_single_sample_forward)
{
    std::mt19937 rng(1);
    my_torch::Network network = random_network({5, 4, 3});
    my_torch::Matrix batch = random_matrix(6, 5, rng);
    my_torch::Matrix outputs = network(batch);

    for (std::size_t row = 0; row < 6; row++)
        assert_matrix_eq(row_of(outputs, row), network(row_of(batch, row)));
}

Test(network, softmax_output_gives_probabilities)
{
    my_torch::Network network(layers_of(hidden_layer(), output_layer()));

    assert_matrix_eq(network(my_torch::Matrix({{1, 2}, {0, 0}, {3, -1}})).row_sums(), my_torch::Matrix({{1}, {1}, {1}}));
}

Test(network, shape_comes_from_the_first_and_last_sized_layers)
{
    my_torch::Network network = random_network({768, 256, 64, 3});

    cr_assert(network.shape() == my_torch::Shape(768, 3));
}

Test(network, keeps_its_layers_in_order)
{
    std::unique_ptr<my_torch::Layer> hidden = hidden_layer();
    std::unique_ptr<my_torch::Layer> output = output_layer();
    my_torch::Layer *first = hidden.get();
    my_torch::Layer *second = output.get();
    my_torch::Network network(layers_of(std::move(hidden), std::move(output)));

    cr_assert_eq(network.layers[0].get(), first);
    cr_assert_eq(network.layers[1].get(), second);
}

Test(network, a_layer_without_weights_can_sit_between_dense_layers)
{
    LayerList layers = layers_of(hidden_layer(), std::make_unique<PassThrough>());
    layers.push_back(output_layer());
    my_torch::Network network(std::move(layers));
    my_torch::Network without(layers_of(hidden_layer(), output_layer()));

    cr_assert(network.shape() == my_torch::Shape(2, 3));
    assert_matrix_eq(network(my_torch::Matrix::row({1, 2})), without(my_torch::Matrix::row({1, 2})));
}

Test(network, rejects_a_network_without_layers)
{
    cr_assert_throw(my_torch::Network(LayerList{}), std::invalid_argument);
}

Test(network, rejects_a_network_without_sized_layers)
{
    cr_assert_throw(my_torch::Network(layers_of(std::make_unique<PassThrough>(), std::make_unique<PassThrough>())), std::invalid_argument);
}

Test(network, rejects_an_empty_layer)
{
    cr_assert_throw(my_torch::Network(layers_of(hidden_layer(), nullptr)), std::invalid_argument);
}

Test(network, rejects_dense_layers_whose_sizes_do_not_chain)
{
    cr_assert_throw(my_torch::Network(layers_of(output_layer(), hidden_layer())), std::invalid_argument);
}
