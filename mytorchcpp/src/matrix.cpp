/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A matrix of doubles stored row by row; nested rows must all have the same size.
** Like numpy, + and - repeat a single row on every row of the other matrix
*/

#include <stdexcept>
#include <string>

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

Matrix Matrix::row(const std::vector<double> &values)
{
    return Matrix(std::vector<std::vector<double>>{values});
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

Matrix Matrix::transpose() const
{
    Matrix result(_cols, _rows);
    for (std::size_t row = 0; row < _rows; row++)
        for (std::size_t col = 0; col < _cols; col++)
            result(col, row) = (*this)(row, col);
    return result;
}

Matrix Matrix::apply(double (*function)(double)) const
{
    Matrix result = *this;
    for (double &value : result._values)
        value = function(value);
    return result;
}

Matrix Matrix::row_sums() const
{
    Matrix result(_rows, 1);
    for (std::size_t row = 0; row < _rows; row++)
        for (std::size_t col = 0; col < _cols; col++)
            result(row, 0) += (*this)(row, col);
    return result;
}

Matrix Matrix::column_sums() const
{
    Matrix result(1, _cols);
    for (std::size_t row = 0; row < _rows; row++)
        for (std::size_t col = 0; col < _cols; col++)
            result(0, col) += (*this)(row, col);
    return result;
}

static bool is_row_of(const Matrix &row, const Matrix &matrix)
{
    return row.rows() == 1 && row.cols() == matrix.cols();
}

static void check_broadcastable(const Matrix &left, const Matrix &right)
{
    if (left.rows() == right.rows() && left.cols() == right.cols())
        return;
    if (is_row_of(right, left))
        return;
    throw std::invalid_argument("element-wise operations need matrices of the same shape or a single row to repeat, got " + std::to_string(left.rows()) + "x" + std::to_string(left.cols()) + " and " + std::to_string(right.rows()) + "x" + std::to_string(right.cols()));
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    check_broadcastable(*this, other);
    for (std::size_t i = 0; i < _values.size(); i++)
        _values[i] += other._values[i % other._values.size()];
    return *this;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
    check_broadcastable(*this, other);
    for (std::size_t i = 0; i < _values.size(); i++)
        _values[i] -= other._values[i % other._values.size()];
    return *this;
}

Matrix &Matrix::operator*=(double scalar)
{
    for (double &value : _values)
        value *= scalar;
    return *this;
}

Matrix &Matrix::operator/=(double scalar)
{
    for (double &value : _values)
        value /= scalar;
    return *this;
}

static void check_product_sizes(const Matrix &left, const Matrix &right)
{
    if (left.cols() != right.rows())
        throw std::invalid_argument("a matrix product needs as many columns on the left (" + std::to_string(left.cols()) + ") as rows on the right (" + std::to_string(right.rows()) + ")");
}

Matrix operator*(const Matrix &left, const Matrix &right)
{
    check_product_sizes(left, right);
    Matrix result(left.rows(), right.cols());
    for (std::size_t i = 0; i < left.rows(); i++)
        for (std::size_t k = 0; k < left.cols(); k++)
            for (std::size_t j = 0; j < right.cols(); j++)
                result(i, j) += left(i, k) * right(k, j);
    return result;
}

Matrix operator+(Matrix left, const Matrix &right)
{
    return left += right;
}

Matrix operator-(Matrix left, const Matrix &right)
{
    return left -= right;
}

Matrix operator*(Matrix matrix, double scalar)
{
    return matrix *= scalar;
}

Matrix operator*(double scalar, Matrix matrix)
{
    return matrix *= scalar;
}

Matrix operator/(Matrix matrix, double scalar)
{
    return matrix /= scalar;
}

}
