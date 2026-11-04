/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Every number is checked: a class must be below n_classes and an input index below n_inputs,
** and errors give the line they come from; empty lines are skipped
*/

#include <limits>
#include <sstream>
#include <vector>

#include "io/boards/boards.hpp"

namespace my_torch {

static std::size_t read_count(std::istringstream &fields, const std::string &where)
{
    std::string text;
    fields >> text;
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos || text.size() > 18)
        throw BoardsError(where + ": expected a whole number, got '" + text + "'");
    return std::stoull(text);
}

BoardsReader::BoardsReader(std::istream &stream)
:
    _stream(stream)
{
    std::string line;
    if (!std::getline(_stream, line))
        throw BoardsError("boards line 1: expected \"n_inputs n_classes\", got nothing");
    std::istringstream fields(line);
    n_inputs = read_count(fields, "boards line 1");
    n_classes = read_count(fields, "boards line 1");
    if (n_inputs < 1 || !(fields >> std::ws).eof())
        throw BoardsError("boards line 1: expected \"n_inputs n_classes\", got '" + line + "'");
}

bool BoardsReader::read_board(Boards &boards, std::size_t row, const std::string &line)
{
    std::string where = "boards line " + std::to_string(_line);
    std::istringstream fields(line);
    std::string label;
    fields >> label;
    bool labelled = label != "-";
    if (labelled) {
        std::istringstream label_field(label);
        std::size_t index = read_count(label_field, where);
        if (index >= n_classes)
            throw BoardsError(where + ": class " + label + " is not below " + std::to_string(n_classes));
        boards.labels(row, index) = 1.0;
    }
    while (!(fields >> std::ws).eof()) {
        std::size_t index = read_count(fields, where);
        if (index >= n_inputs)
            throw BoardsError(where + ": input " + std::to_string(index) + " is not below " + std::to_string(n_inputs));
        boards.inputs(row, index) = 1.0;
    }
    return labelled;
}

Boards BoardsReader::next(std::size_t count)
{
    std::vector<std::string> lines;
    std::string line;
    std::vector<std::size_t> numbers;
    while (lines.size() < count && std::getline(_stream, line)) {
        _line++;
        if (line.find_first_not_of(" \t\r") == std::string::npos)
            continue;
        lines.push_back(line);
        numbers.push_back(_line);
    }
    Boards boards {Matrix(lines.size(), n_inputs), Matrix(lines.size(), n_classes), 0};
    std::size_t last = _line;
    for (std::size_t row = 0; row < lines.size(); row++) {
        _line = numbers[row];
        boards.labelled += read_board(boards, row, lines[row]);
    }
    _line = last;
    return boards;
}

Boards BoardsReader::all()
{
    return next(std::numeric_limits<std::size_t>::max());
}

}
