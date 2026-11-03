/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the .npy reader and writer
*/

#include <bit>
#include <cstdint>
#include <criterion/criterion.h>
#include <string>

#include "io/npy/npy.hpp"

using my_torch::Matrix;
using my_torch::NpyError;

static std::string header_of(const std::string &bytes)
{
    return bytes.substr(10, static_cast<unsigned char>(bytes[8]) | static_cast<unsigned char>(bytes[9]) << 8);
}

Test(npy, starts_like_a_numpy_file_with_data_aligned_on_64_bytes)
{
    std::string bytes = my_torch::to_npy(Matrix({{1, 2, 3}, {4, 5, 6}}));
    std::string header = header_of(bytes);

    cr_assert_eq(bytes.compare(0, 8, std::string("\x93NUMPY\x01\x00", 8)), 0);
    cr_assert_eq((10 + header.size()) % 64, 0);
    cr_assert_eq(header.back(), '\n');
    cr_assert_eq(header.find("{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }"), 0);
    cr_assert_eq(bytes.size(), 10 + header.size() + 6 * 8);
}

Test(npy, saves_a_row_as_a_vector)
{
    std::string bytes = my_torch::to_npy(Matrix::row({1, 2, 3}), true);

    cr_assert_neq(header_of(bytes).find("'shape': (3,)"), std::string::npos);
    cr_assert_throw(my_torch::to_npy(Matrix(2, 2), true), NpyError);
}

Test(npy, values_are_little_endian_doubles_row_by_row)
{
    std::string bytes = my_torch::to_npy(Matrix({{1.5, -2}}));
    std::string data = bytes.substr(bytes.size() - 16);
    std::uint64_t first = 0;

    for (int index = 7; index >= 0; index--)
        first = first << 8 | static_cast<unsigned char>(data[index]);
    cr_assert_eq(std::bit_cast<double>(first), 1.5);
}

Test(npy, reads_back_bit_for_bit)
{
    Matrix matrix({{0.1, 1.0 / 3.0, -0.0}, {4.9e-324, 1e308, -7.25}});
    Matrix read = my_torch::from_npy(my_torch::to_npy(matrix));

    cr_assert_eq(read.rows(), 2);
    cr_assert_eq(read.cols(), 3);
    for (std::size_t row = 0; row < 2; row++)
        for (std::size_t col = 0; col < 3; col++)
            cr_assert_eq(std::bit_cast<std::uint64_t>(read(row, col)), std::bit_cast<std::uint64_t>(matrix(row, col)));
}

Test(npy, a_vector_reads_back_as_one_row)
{
    Matrix read = my_torch::from_npy(my_torch::to_npy(Matrix::row({1, 2, 3, 4}), true));

    cr_assert_eq(read.rows(), 1);
    cr_assert_eq(read.cols(), 4);
    cr_assert_eq(read(0, 3), 4);
}

Test(npy, rejects_what_is_not_a_float64_array)
{
    std::string bytes = my_torch::to_npy(Matrix({{1, 2}, {3, 4}}));
    std::string as_float32 = bytes;
    std::string fortran = bytes;
    std::string version_2 = bytes;
    std::string three_dimensions = bytes;

    as_float32.replace(as_float32.find("<f8"), 3, "<f4");
    fortran.replace(fortran.find("False"), 5, "True ");
    version_2[6] = '\x02';
    three_dimensions.replace(three_dimensions.find("(2, 2)"), 6, "(1,2,2)");
    for (const std::string &bad : {std::string("not numpy"), std::string(), as_float32, fortran, version_2, three_dimensions,
             bytes.substr(0, bytes.size() - 1), bytes + "x", bytes.substr(0, 20)})
        cr_assert_throw(my_torch::from_npy(bad), NpyError);
}
