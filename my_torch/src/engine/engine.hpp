/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** The engine is how the Python analyzer uses the C++ network: it reads boards encoded as input
** indices (io/boards) and either prints the network's outputs (predict) or trains it with early
** stopping and saves it (train). A network saved without a training starts from
** default_training, the configuration the hyperparameter search retained
*/

#pragma once

#include <istream>
#include <ostream>
#include <string>
#include <vector>

#include "training/training.hpp"

namespace my_torch {

inline constexpr std::size_t PREDICT_CHUNK = 1024;

Training default_training();
void predict(const std::string &network_path, std::istream &boards, std::ostream &out);
void train(const std::string &network_path, const std::string &save_path, std::istream &boards, std::ostream &out);
int run_engine(const std::vector<std::string> &arguments, std::istream &in, std::ostream &out, std::ostream &err);

}
