/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Saves a network to a .nn file and loads it back, with free metadata and, when given, its
** training so it can be resumed. Any problem with a file raises NetworkFileError, which the
** analyzer prints on stderr before exiting with 84
*/

#pragma once

#include <optional>
#include <stdexcept>
#include <string>

#include "io/json/json.hpp"
#include "model/network/network.hpp"
#include "training/training.hpp"

namespace my_torch {

inline constexpr const char *NETWORK_FORMAT = "my_torch";
inline constexpr int NETWORK_FORMAT_VERSION = 1;

class NetworkFileError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct SavedNetwork {
    Network network;
    Json metadata;
    std::optional<Training> training;
};

std::unique_ptr<Loss> build_loss(const Json &description);
std::unique_ptr<Optimizer> build_optimizer(const Json &description);
void save_network(const std::string &path, Network &network, const Json &metadata = Json(Json::Object()), const Training *training = nullptr);
SavedNetwork load_network(const std::string &path);

}
