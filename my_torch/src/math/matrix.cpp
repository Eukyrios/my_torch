/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A matrix of doubles stored row by row; nested rows must all have the same size.
** Like numpy, element-wise operations repeat a single row or column over the other matrix
*/

#include <algorithm>
#include <stdexcept>
#include <string>

#include "math/matrix.hpp"

namespace my_torch {

static void check_rows_have_the_same_size(const std::vector<std::vector<double>> &values)
{
    for (const std::vector<double> &row : values)
        if (row.size() != values[0].size())
            throw std::invalid_argument("a matrix needs rows of the same size");
}

Matrix::Matrix(std::size_t rows, std::size_t cols, double value)
:
    _rows(rows), _cols(cols), _values(rows * cols, value)
{}

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

std::size_t Matrix::size() const
{
    return _values.size();
}

double &Matrix::operator()(std::size_t row, std::size_t col)
{
    return _values[row * _cols + col];
}

double Matrix::operator()(std::size_t row, std::size_t col) const
{
    return _values[row * _cols + col];
}

static void check_product_sizes(const Matrix &left, const Matrix &right)
{
    if (left.cols() != right.rows())
        throw std::invalid_argument("a matrix product needs as many columns on the left (" + std::to_string(left.cols()) + ") as rows on the right (" + std::to_string(right.rows()) + ")");
}

Matrix Matrix::matmul(const Matrix &other) const
{
    check_product_sizes(*this, other);
    Matrix result(_rows, other._cols);
    for (std::size_t i = 0; i < _rows; i++)
        for (std::size_t k = 0; k < _cols; k++)
            for (std::size_t j = 0; j < other._cols; j++)
                result(i, j) += (*this)(i, k) * other(k, j);
    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(_cols, _rows);
    for (std::size_t row = 0; row < _rows; row++)
        for (std::size_t col = 0; col < _cols; col++)
            result(col, row) = (*this)(row, col);
    return result;
}

Matrix Matrix::select_rows(const std::vector<std::size_t> &indices) const
{
    Matrix result(indices.size(), _cols);
    for (std::size_t row = 0; row < indices.size(); row++) {
        if (indices[row] >= _rows)
            throw std::out_of_range("row " + std::to_string(indices[row]) + " is outside a matrix of " + std::to_string(_rows) + " rows");
        for (std::size_t col = 0; col < _cols; col++)
            result(row, col) = (*this)(indices[row], col);
    }
    return result;
}

Matrix Matrix::apply(double (*function)(double)) const
{
    Matrix result = *this;
    for (double &value : result._values)
        value = function(value);
    return result;
}

double Matrix::sum() const
{
    double total = 0.0;
    for (double value : _values)
        total += value;
    return total;
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

Matrix Matrix::row_maxima() const
{
    Matrix result(_rows, 1);
    for (std::size_t row = 0; row < _rows; row++) {
        result(row, 0) = (*this)(row, 0);
        for (std::size_t col = 1; col < _cols; col++)
            result(row, 0) = std::max(result(row, 0), (*this)(row, col));
    }
    return result;
}

std::vector<std::size_t> Matrix::row_argmax() const
{
    std::vector<std::size_t> indices(_rows, 0);
    for (std::size_t row = 0; row < _rows; row++)
        for (std::size_t col = 1; col < _cols; col++)
            if ((*this)(row, col) > (*this)(row, indices[row]))
                indices[row] = col;
    return indices;
}

static bool is_row_of(const Matrix &row, const Matrix &matrix)
{
    return row.rows() == 1 && row.cols() == matrix.cols();
}

static bool is_column_of(const Matrix &column, const Matrix &matrix)
{
    return column.cols() == 1 && column.rows() == matrix.rows();
}

static void check_broadcastable(const Matrix &left, const Matrix &right)
{
    if (left.rows() == right.rows() && left.cols() == right.cols())
        return;
    if (is_row_of(right, left) || is_column_of(right, left))
        return;
    throw std::invalid_argument("element-wise operations need matrices of the same shape, or a single row or column to repeat, got " + std::to_string(left.rows()) + "x" + std::to_string(left.cols()) + " and " + std::to_string(right.rows()) + "x" + std::to_string(right.cols()));
}

static double broadcast_value(const Matrix &other, std::size_t row, std::size_t col)
{
    return other(other.rows() == 1 ? 0 : row, other.cols() == 1 ? 0 : col);
}

Matrix &Matrix::operator+=(const Matrix &other)
{
    check_broadcastable(*this, other);
    for (std::size_t row = 0; row < _rows; row++)
        for (std::size_t col = 0; col < _cols; col++)
            (*this)(row, col) += broadcast_value(other, row, col);
    return *this;
}

Matrix &Matrix::operator-=(const Matrix &other)
{
    check_broadcastable(*this, other);
    for (std::size_t row = 0; row < _rows; row++)
        for (std::size_t col = 0; col < _cols; col++)
            (*this)(row, col) -= broadcast_value(other, row, col);
    return *this;
}

Matrix &Matrix::operator*=(const Matrix &other)
{
    check_broadcastable(*this, other);
    for (std::size_t row = 0; row < _rows; row++)
        for (std::size_t col = 0; col < _cols; col++)
            (*this)(row, col) *= broadcast_value(other, row, col);
    return *this;
}

Matrix &Matrix::operator/=(const Matrix &other)
{
    check_broadcastable(*this, other);
    for (std::size_t row = 0; row < _rows; row++)
        for (std::size_t col = 0; col < _cols; col++)
            (*this)(row, col) /= broadcast_value(other, row, col);
    return *this;
}

Matrix Matrix::clip(double low, double high) const
{
    Matrix result = *this;
    for (double &value : result._values)
        value = std::clamp(value, low, high);
    return result;
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

Matrix operator*(Matrix left, const Matrix &right)
{
    return left *= right;
}

Matrix operator/(Matrix left, const Matrix &right)
{
    return left /= right;
}

}
