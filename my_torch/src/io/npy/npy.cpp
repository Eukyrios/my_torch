/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A .npy file is the magic "\x93NUMPY", the version 1.0, the header length on 2 bytes, then a
** Python dict giving the type, the order and the shape, padded so the data starts on 64 bytes;
** the values follow row by row as little-endian doubles. A vector (n,) reads as a 1 x n row
*/

#include <bit>
#include <cstdint>
#include <vector>

#include "io/npy/npy.hpp"

namespace my_torch {

static const std::string MAGIC = "\x93NUMPY";
static const std::size_t PREFIX_SIZE = MAGIC.size() + 4;
static const std::size_t ALIGNMENT = 64;

static std::string little_endian(std::uint64_t value, std::size_t size)
{
    std::string bytes;
    for (std::size_t index = 0; index < size; index++)
        bytes += static_cast<char>((value >> (8 * index)) & 0xFF);
    return bytes;
}

static std::uint64_t read_little_endian(const std::string &bytes, std::size_t offset, std::size_t size)
{
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < size; index++)
        value |= static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[offset + index])) << (8 * index);
    return value;
}

static std::string shape_text(const Matrix &matrix, bool as_vector)
{
    if (as_vector)
        return "(" + std::to_string(matrix.size()) + ",)";
    return "(" + std::to_string(matrix.rows()) + ", " + std::to_string(matrix.cols()) + ")";
}

static std::string header(const Matrix &matrix, bool as_vector)
{
    std::string dict = "{'descr': '<f8', 'fortran_order': False, 'shape': " + shape_text(matrix, as_vector) + ", }";
    std::size_t padding = ALIGNMENT - (PREFIX_SIZE + dict.size() + 1) % ALIGNMENT;
    return dict + std::string(padding % ALIGNMENT, ' ') + "\n";
}

std::string to_npy(const Matrix &matrix, bool as_vector)
{
    if (as_vector && matrix.rows() > 1)
        throw NpyError("only a single row can be saved as a vector, got " + std::to_string(matrix.rows()) + " rows");
    std::string dict = header(matrix, as_vector);
    std::string bytes = MAGIC + '\x01' + '\x00' + little_endian(dict.size(), 2) + dict;
    for (std::size_t row = 0; row < matrix.rows(); row++)
        for (std::size_t col = 0; col < matrix.cols(); col++)
            bytes += little_endian(std::bit_cast<std::uint64_t>(matrix(row, col)), 8);
    return bytes;
}

/* Reading */

static std::string header_value(const std::string &dict, const std::string &key)
{
    std::size_t start = dict.find("'" + key + "':");
    if (start == std::string::npos)
        throw NpyError("the header has no '" + key + "'");
    start = dict.find_first_not_of(' ', start + key.size() + 3);
    if (start == std::string::npos)
        throw NpyError("the header has an invalid '" + key + "'");
    std::size_t end = dict[start] == '(' ? dict.find(')', start) + 1 : dict.find_first_of(",}", start);
    if (end == std::string::npos || end == 0)
        throw NpyError("the header has an invalid '" + key + "'");
    return dict.substr(start, end - start);
}

static std::vector<std::size_t> parse_shape(const std::string &text)
{
    std::vector<std::size_t> shape;
    std::size_t position = 1;
    while (position < text.size() - 1) {
        position = text.find_first_not_of(", ", position);
        if (position >= text.size() - 1)
            break;
        std::size_t end = text.find_first_not_of("0123456789", position);
        if (end == position)
            throw NpyError("invalid shape " + text);
        shape.push_back(std::stoull(text.substr(position, end - position)));
        position = end;
    }
    if (shape.empty() || shape.size() > 2)
        throw NpyError("expected a vector or a matrix, got shape " + text);
    return shape;
}

static std::string read_header(const std::string &bytes)
{
    if (bytes.size() < PREFIX_SIZE || bytes.compare(0, MAGIC.size(), MAGIC) != 0)
        throw NpyError("not a .npy array");
    if (bytes[MAGIC.size()] != '\x01')
        throw NpyError("unsupported .npy version " + std::to_string(static_cast<int>(bytes[MAGIC.size()])));
    std::size_t length = read_little_endian(bytes, MAGIC.size() + 2, 2);
    if (bytes.size() < PREFIX_SIZE + length)
        throw NpyError("truncated .npy header");
    return bytes.substr(PREFIX_SIZE, length);
}

Matrix from_npy(const std::string &bytes)
{
    std::string dict = read_header(bytes);
    if (header_value(dict, "descr") != "'<f8'")
        throw NpyError("expected float64 values ('<f8'), got " + header_value(dict, "descr"));
    if (header_value(dict, "fortran_order") != "False")
        throw NpyError("expected values row by row (fortran_order False)");
    std::vector<std::size_t> shape = parse_shape(header_value(dict, "shape"));
    std::size_t rows = shape.size() == 2 ? shape[0] : 1;
    std::size_t cols = shape.back();
    std::size_t offset = PREFIX_SIZE + dict.size();
    if (bytes.size() != offset + rows * cols * 8)
        throw NpyError("expected " + std::to_string(rows * cols) + " values, got " + std::to_string((bytes.size() - offset) / 8));
    Matrix matrix(rows, cols);
    for (std::size_t row = 0; row < rows; row++)
        for (std::size_t col = 0; col < cols; col++)
            matrix(row, col) = std::bit_cast<double>(read_little_endian(bytes, offset + 8 * (row * cols + col), 8));
    return matrix;
}

}
