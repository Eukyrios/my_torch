/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** my_torch_generator: writes nb blank networks for each config file given, see run_generator
*/

#include <iostream>
#include <string>
#include <vector>

#include "generator/generator.hpp"

int main(int argc, char **argv)
{
    return my_torch::run_generator(std::vector<std::string>(argv + 1, argv + argc), std::cout, std::cerr);
}
