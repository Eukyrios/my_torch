/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A matrix of doubles stored row by row, what numpy arrays were in the Python version
*/

#pragma once

#include <cstddef>
#include <vector>

namespace my_torch {

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols, double value = 0.0);
    Matrix(const std::vector<std::vector<double>> &values);

    std::size_t rows() const;
    std::size_t cols() const;
    double &operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;

private:
    std::size_t _rows = 0;
    std::size_t _cols = 0;
    std::vector<double> _values;
};

Matrix operator*(const Matrix &left, const Matrix &right);

}
