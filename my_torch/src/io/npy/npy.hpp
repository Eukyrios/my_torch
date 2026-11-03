/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** numpy's .npy format for one array of float64: a matrix is saved with its shape (rows, cols),
** or as a vector (n,) for a single row such as biases, and read back bit for bit
*/

#pragma once

#include <stdexcept>
#include <string>

#include "math/matrix.hpp"

namespace my_torch {

class NpyError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::string to_npy(const Matrix &matrix, bool as_vector = false);
Matrix from_npy(const std::string &bytes);

}
