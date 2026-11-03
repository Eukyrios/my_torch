/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** my_torch_engine, run by the Python analyzer: see src/engine/engine.hpp
*/

#include <iostream>
#include <string>
#include <vector>

#include "engine/engine.hpp"

int main(int argc, char **argv)
{
    std::ios::sync_with_stdio(false);
    return my_torch::run_engine(std::vector<std::string>(argv + 1, argv + argc), std::cin, std::cout, std::cerr);
}
