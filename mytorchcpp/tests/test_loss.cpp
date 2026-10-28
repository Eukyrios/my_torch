/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the losses
*/

#include <criterion/criterion.h>
#include <stdexcept>

#include "assert_matrix.hpp"
#include "loss.hpp"

static const my_torch::Matrix PREDICTIONS({{0.7, 0.2, 0.1}, {0.1, 0.8, 0.1}});
static const my_torch::Matrix EXPECTED({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}});

static my_torch::Matrix numerical_gradient(const my_torch::Loss &loss, const my_torch::Matrix &predictions, const my_torch::Matrix &expected)
{
    const double step = 1e-6;
    my_torch::Matrix gradient(predictions.rows(), predictions.cols());
    for (std::size_t row = 0; row < predictions.rows(); row++)
        for (std::size_t col = 0; col < predictions.cols(); col++) {
            my_torch::Matrix higher = predictions;
            my_torch::Matrix lower = predictions;
            higher(row, col) += step;
            lower(row, col) -= step;
            gradient(row, col) = (loss(higher, expected) - loss(lower, expected)) / (2 * step);
        }
    return gradient;
}

static void assert_close(const my_torch::Matrix &actual, const my_torch::Matrix &expected, double tolerance)
{
    cr_assert_eq(actual.rows(), expected.rows());
    cr_assert_eq(actual.cols(), expected.cols());
    for (std::size_t row = 0; row < expected.rows(); row++)
        for (std::size_t col = 0; col < expected.cols(); col++)
            cr_assert_float_eq(actual(row, col), expected(row, col), tolerance);
}

Test(mean_squared_error, matches_the_hand_computation)
{
    cr_assert_float_eq(my_torch::MeanSquaredError()(my_torch::Matrix::row({1.0, 2.0}), my_torch::Matrix::row({0.0, 4.0})), 2.5, 1e-12);
}

Test(mean_squared_error, is_zero_for_perfect_predictions)
{
    cr_assert_float_eq(my_torch::MeanSquaredError()(PREDICTIONS, PREDICTIONS), 0.0, 1e-12);
}

Test(mean_squared_error, gradient_matches_the_numerical_gradient)
{
    my_torch::MeanSquaredError loss;

    assert_close(loss.gradient(PREDICTIONS, EXPECTED), numerical_gradient(loss, PREDICTIONS, EXPECTED), 1e-6);
}

Test(mean_squared_error, is_averaged_over_the_batch)
{
    my_torch::MeanSquaredError loss;
    my_torch::Matrix one_prediction = my_torch::Matrix::row({0.7, 0.2, 0.1});
    my_torch::Matrix one_expected = my_torch::Matrix::row({1.0, 0.0, 0.0});
    my_torch::Matrix two_predictions({{0.7, 0.2, 0.1}, {0.7, 0.2, 0.1}});
    my_torch::Matrix two_expected({{1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}});

    cr_assert_float_eq(loss(two_predictions, two_expected), loss(one_prediction, one_expected), 1e-12);
}

Test(mean_squared_error, gradient_has_the_shape_of_the_predictions)
{
    my_torch::Matrix gradient = my_torch::MeanSquaredError().gradient(PREDICTIONS, EXPECTED);

    cr_assert_eq(gradient.rows(), 2u);
    cr_assert_eq(gradient.cols(), 3u);
}

Test(mean_squared_error, rejects_predictions_and_expected_of_different_shapes)
{
    cr_assert_throw(my_torch::MeanSquaredError()(PREDICTIONS, my_torch::Matrix(2, 2)), std::invalid_argument);
    cr_assert_throw(my_torch::MeanSquaredError().gradient(PREDICTIONS, my_torch::Matrix(2, 2)), std::invalid_argument);
}

Test(mean_squared_error, has_its_name)
{
    my_torch::MeanSquaredError loss;

    cr_assert_str_eq(loss.name.c_str(), "mean_squared_error");
}
