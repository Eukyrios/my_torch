/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A small network learns XOR end to end, the first problem a single neuron cannot solve
*/

#include <criterion/criterion.h>

#include "model/activations/activations/softmax/softmax.hpp"
#include "model/activations/activations/tanh/tanh.hpp"
#include "model/layers/layers/dense_layer/dense_layer.hpp"
#include "training/losses/losses/cross_entropy/cross_entropy.hpp"
#include "model/network/network.hpp"
#include "training/optimizers/optimizers/sgd/sgd.hpp"

static const my_torch::Matrix XOR_INPUTS({{0, 0}, {0, 1}, {1, 0}, {1, 1}});
static const my_torch::Matrix XOR_LABELS({{1, 0}, {0, 1}, {0, 1}, {1, 0}});
static const std::vector<std::size_t> XOR_OUTPUTS = {0, 1, 1, 0};

struct TrainedXor {
    my_torch::Network network;
    std::vector<double> losses;
};

static TrainedXor train_xor(unsigned int seed)
{
    std::mt19937 rng(seed);
    std::vector<std::unique_ptr<my_torch::Layer>> layers;
    layers.push_back(my_torch::DenseLayer::create(2, 8, std::make_unique<my_torch::Tanh>(), rng));
    layers.push_back(my_torch::DenseLayer::create(8, 2, std::make_unique<my_torch::Softmax>(), rng));
    my_torch::Network network(std::move(layers));
    my_torch::SGD optimizer(0.5);
    std::vector<double> losses = network.train(XOR_INPUTS, XOR_LABELS, my_torch::CrossEntropy(), optimizer, 500, 4, rng);
    return TrainedXor{std::move(network), losses};
}

Test(xor, the_network_learns_xor)
{
    TrainedXor trained = train_xor(0);

    cr_assert(trained.network(XOR_INPUTS).row_argmax() == XOR_OUTPUTS);
    cr_assert_lt(trained.losses.back(), 0.05);
    cr_assert_gt(trained.losses.front(), 0.05);
}

Test(xor, xor_is_learned_from_several_random_starts)
{
    for (unsigned int seed = 0; seed < 5; seed++) {
        TrainedXor trained = train_xor(seed);
        cr_assert(trained.network(XOR_INPUTS).row_argmax() == XOR_OUTPUTS, "seed %u", seed);
    }
}
