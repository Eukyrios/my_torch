/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Builds blank networks from a JSON configuration (inputs, layers, training, metadata), in code with
** generate_network or as .nn files with generate_files; network i of a file uses seed + i - 1
*/

#pragma once

#include <cstddef>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/json/json.hpp"
#include "model/network/network.hpp"
#include "training/training.hpp"

namespace my_torch {

class ConfigError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct GeneratedNetwork {
    Network network;
    Training training;
    Json metadata;
};

Json read_config(const std::string &path);
GeneratedNetwork generate_network(const Json &config, unsigned int seed);
std::string generated_name(const std::string &config_path, std::size_t number);
std::vector<std::string> generate_files(const std::string &config_path, std::size_t count, const std::string &out_dir = ".");
int run_generator(const std::vector<std::string> &arguments, std::ostream &out, std::ostream &err);

}
