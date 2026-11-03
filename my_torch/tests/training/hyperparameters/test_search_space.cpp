/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the hyperparameter search space
*/

#include <cmath>
#include <criterion/criterion.h>
#include <stdexcept>

#include "training/hyperparameters/search_space.hpp"

static my_torch::SearchSpace space()
{
    return my_torch::SearchSpace({
        {"learning_rate", 1e-4, 1.0, true, false},
        {"momentum", 0.0, 0.9, false, false},
        {"batch_size", 8, 256, true, true},
    });
}

Test(search_space, decodes_the_corners_to_the_bounds)
{
    my_torch::Hyperparameters low = space().decode({0, 0, 0});
    my_torch::Hyperparameters high = space().decode({1, 1, 1});

    cr_assert_float_eq(low.at("learning_rate"), 1e-4, 1e-15);
    cr_assert_float_eq(low.at("momentum"), 0.0, 1e-15);
    cr_assert_float_eq(low.at("batch_size"), 8.0, 1e-15);
    cr_assert_float_eq(high.at("learning_rate"), 1.0, 1e-12);
    cr_assert_float_eq(high.at("momentum"), 0.9, 1e-15);
    cr_assert_float_eq(high.at("batch_size"), 256.0, 1e-12);
}

Test(search_space, a_linear_scale_maps_the_middle_to_the_middle)
{
    cr_assert_float_eq(space().decode({0, 0.5, 0}).at("momentum"), 0.45, 1e-15);
}

Test(search_space, a_log_scale_maps_the_middle_to_the_geometric_middle)
{
    cr_assert_float_eq(space().decode({0.5, 0, 0}).at("learning_rate"), 1e-2, 1e-12);
    cr_assert_float_eq(space().decode({0.25, 0, 0}).at("learning_rate"), 1e-3, 1e-12);
}

Test(search_space, whole_dimensions_are_rounded)
{
    double batch_size = space().decode({0, 0, 0.5}).at("batch_size");

    cr_assert_float_eq(batch_size, std::round(batch_size), 1e-15);
    cr_assert_float_eq(batch_size, 45.0, 1e-12);
}

Test(search_space, coordinates_outside_the_cube_are_kept_within_the_bounds)
{
    my_torch::Hyperparameters values = space().decode({-3, 7, 1.5});

    cr_assert_float_eq(values.at("learning_rate"), 1e-4, 1e-15);
    cr_assert_float_eq(values.at("momentum"), 0.9, 1e-15);
    cr_assert_float_eq(values.at("batch_size"), 256.0, 1e-12);
}

Test(search_space, random_positions_are_in_the_unit_cube_and_reproducible)
{
    std::mt19937 first(3);
    std::mt19937 second(3);
    std::vector<double> position = space().random_position(first);

    cr_assert_eq(position.size(), 3u);
    for (double coordinate : position)
        cr_assert(coordinate >= 0.0 && coordinate <= 1.0);
    cr_assert(position == space().random_position(second));
}

Test(search_space, decode_needs_one_coordinate_per_dimension)
{
    cr_assert_throw(space().decode({0.5, 0.5}), std::invalid_argument);
}

Test(search_space, rejects_invalid_dimensions)
{
    cr_assert_throw(my_torch::SearchSpace({}), std::invalid_argument);
    cr_assert_throw(my_torch::SearchSpace({{"", 0, 1}}), std::invalid_argument);
    cr_assert_throw(my_torch::SearchSpace({{"rate", 1, 1}}), std::invalid_argument);
    cr_assert_throw(my_torch::SearchSpace({{"rate", 0, 1, true}}), std::invalid_argument);
    cr_assert_throw(my_torch::SearchSpace({{"rate", 0, 1}, {"rate", 0, 2}}), std::invalid_argument);
}
