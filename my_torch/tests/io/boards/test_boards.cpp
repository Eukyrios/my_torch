/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the reader of boards encoded as input indices
*/

#include <criterion/criterion.h>
#include <sstream>

#include "assert_matrix.hpp"
#include "io/boards/boards.hpp"

Test(boards, reads_the_classes_and_the_inputs_that_are_1)
{
    std::istringstream text("4 3\n2 0 3\n0 1\n\n1\n");
    my_torch::BoardsReader reader(text);
    my_torch::Boards boards = reader.all();

    cr_assert_eq(reader.n_inputs, 4);
    cr_assert_eq(reader.n_classes, 3);
    cr_assert_eq(boards.labelled, 3);
    assert_matrix_eq(boards.inputs, my_torch::Matrix({{1, 0, 0, 1}, {0, 1, 0, 0}, {0, 0, 0, 0}}));
    assert_matrix_eq(boards.labels, my_torch::Matrix({{0, 0, 1}, {1, 0, 0}, {0, 1, 0}}));
}

Test(boards, a_dash_is_a_board_without_class)
{
    std::istringstream text("3 2\n- 0 2\n- 1\n");
    my_torch::Boards boards = my_torch::BoardsReader(text).all();

    cr_assert_eq(boards.labelled, 0);
    assert_matrix_eq(boards.inputs, my_torch::Matrix({{1, 0, 1}, {0, 1, 0}}));
    assert_matrix_eq(boards.labels, my_torch::Matrix(2, 2));
}

Test(boards, reads_a_few_boards_at_a_time)
{
    std::istringstream text("2 2\n0 0\n1 1\n\n0 0 1\n");
    my_torch::BoardsReader reader(text);

    cr_assert_eq(reader.next(2).inputs.rows(), 2);
    my_torch::Boards last = reader.next(2);
    cr_assert_eq(last.inputs.rows(), 1);
    assert_matrix_eq(last.inputs, my_torch::Matrix({{1, 1}}));
    cr_assert_eq(reader.next(2).inputs.rows(), 0);
}

static std::string error_of(const std::string &text)
{
    std::istringstream stream(text);
    try {
        my_torch::BoardsReader(stream).all();
    } catch (const my_torch::BoardsError &error) {
        return error.what();
    }
    return "no error";
}

Test(boards, refuses_bad_numbers_and_says_on_which_line)
{
    cr_assert_eq(error_of(""), std::string("boards line 1: expected \"n_inputs n_classes\", got nothing"));
    cr_assert_eq(error_of("769\n"), std::string("boards line 1: expected a whole number, got ''"));
    cr_assert_eq(error_of("769 5 1\n"), std::string("boards line 1: expected \"n_inputs n_classes\", got '769 5 1'"));
    cr_assert_eq(error_of("0 5\n"), std::string("boards line 1: expected \"n_inputs n_classes\", got '0 5'"));
    cr_assert_eq(error_of("4 2\n0 1\n\n2 1\n"), std::string("boards line 4: class 2 is not below 2"));
    cr_assert_eq(error_of("4 2\n0 4\n"), std::string("boards line 2: input 4 is not below 4"));
    cr_assert_eq(error_of("4 2\n0 -1\n"), std::string("boards line 2: expected a whole number, got '-1'"));
    cr_assert_eq(error_of("4 2\nx 1\n"), std::string("boards line 2: expected a whole number, got 'x'"));
}
