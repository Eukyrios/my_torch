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
    explicit Matrix(std::size_t rows, std::size_t cols, double value = 0.0);
    Matrix(const std::vector<std::vector<double>> &values);

    static Matrix row(const std::vector<double> &values);

    std::size_t rows() const;
    std::size_t cols() const;
    std::size_t size() const;
    double &operator()(std::size_t row, std::size_t col);
    double operator()(std::size_t row, std::size_t col) const;
    Matrix transpose() const;
    Matrix apply(double (*function)(double)) const;
    Matrix hadamard(const Matrix &other) const;
    Matrix clip(double low, double high) const;
    double sum() const;
    Matrix row_sums() const;
    Matrix column_sums() const;
    Matrix row_maxima() const;
    Matrix &operator+=(const Matrix &other);
    Matrix &operator-=(const Matrix &other);
    Matrix &operator*=(double scalar);
    Matrix &operator/=(double scalar);
    Matrix &operator/=(const Matrix &other);

private:
    std::size_t _rows = 0;
    std::size_t _cols = 0;
    std::vector<double> _values;
};

Matrix operator*(const Matrix &left, const Matrix &right);
Matrix operator+(Matrix left, const Matrix &right);
Matrix operator-(Matrix left, const Matrix &right);
Matrix operator*(Matrix matrix, double scalar);
Matrix operator*(double scalar, Matrix matrix);
Matrix operator/(Matrix matrix, double scalar);
Matrix operator/(Matrix left, const Matrix &right);

}
