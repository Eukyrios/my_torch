/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** An uncompressed zip archive (the container of numpy's .npz): named files written in order and
** read back by name, each one checked against its CRC-32 so a damaged archive is refused
*/

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace my_torch {

class ZipError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

using ZipEntries = std::vector<std::pair<std::string, std::string>>;

std::uint32_t crc32(const std::string &bytes);
std::string to_zip(const ZipEntries &entries);
ZipEntries from_zip(const std::string &bytes);

}
