/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A training: how it goes (loss, optimizer with its learning rate and schedule, batch size)
** and where it stands (epochs done, their losses, the random generator shuffling the batches).
** run trains a few more epochs and carries on from there, so it can be saved with the network
** and resumed: 10 epochs, save, load, then 10 more give the same network as 20 in one go
*/

#pragma once

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

#include "math/matrix.hpp"
#include "model/network/network.hpp"
#include "training/losses/loss.hpp"
#include "training/optimizers/optimizer.hpp"

namespace my_torch {

class Training {
public:
    Training(std::unique_ptr<Loss> loss, std::unique_ptr<Optimizer> optimizer, std::size_t batch_size = 32, std::mt19937 rng = std::mt19937());

    std::vector<double> run(Network &network, const Matrix &inputs, const Matrix &labels, std::size_t epochs);

    std::unique_ptr<Loss> loss;
    std::unique_ptr<Optimizer> optimizer;
    std::size_t batch_size;
    std::mt19937 rng;
    std::size_t epoch = 0;
    std::vector<double> losses;
};

}
