/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A matrix of doubles stored row by row; nested rows must all have the same size
*/

#include <stdexcept>

#include "matrix.hpp"

namespace my_torch {

static void check_rows_have_the_same_size(const std::vector<std::vector<double>> &values)
{
    for (const std::vector<double> &row : values)
        if (row.size() != values[0].size())
            throw std::invalid_argument("a matrix needs rows of the same size");
}

Matrix::Matrix(std::size_t rows, std::size_t cols, double value) : _rows(rows), _cols(cols), _values(rows * cols, value)
{
}

Matrix::Matrix(const std::vector<std::vector<double>> &values)
{
    if (values.empty())
        return;
    check_rows_have_the_same_size(values);
    _rows = values.size();
    _cols = values[0].size();
    for (const std::vector<double> &row : values)
        _values.insert(_values.end(), row.begin(), row.end());
}

std::size_t Matrix::rows() const
{
    return _rows;
}

std::size_t Matrix::cols() const
{
    return _cols;
}

double &Matrix::operator()(std::size_t row, std::size_t col)
{
    return _values[row * _cols + col];
}

double Matrix::operator()(std::size_t row, std::size_t col) const
{
    return _values[row * _cols + col];
}

}
