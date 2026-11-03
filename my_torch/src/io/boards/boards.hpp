/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Reads boards encoded as input indices, the format of scripts/encode_dataset.py: a first line
** "n_inputs n_classes", then one board per line, its class index ("-" when unknown) followed by
** the indices of its inputs that are 1, the others being 0. next(count) reads the boards a few at
** a time, so a big file never has to fit in memory as one dense matrix
*/

#pragma once

#include <cstddef>
#include <istream>
#include <stdexcept>
#include <string>

#include "math/matrix.hpp"

namespace my_torch {

class BoardsError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

struct Boards {
    Matrix inputs;          // one row per board
    Matrix labels;          // one one-hot row per board, all zeros when its class is unknown
    std::size_t labelled;   // how many boards have a class
};

class BoardsReader {
public:
    explicit BoardsReader(std::istream &stream);

    Boards next(std::size_t count);
    Boards all();

    std::size_t n_inputs;
    std::size_t n_classes;

private:
    bool read_board(Boards &boards, std::size_t row, const std::string &line);

    std::istream &_stream;
    std::size_t _line = 1;
};

}
