/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the zip reader and writer
*/

#include <criterion/criterion.h>
#include <string>

#include "io/zip/zip.hpp"

using my_torch::ZipEntries;
using my_torch::ZipError;

static const ZipEntries ENTRIES = {
    {"header.json", "{\"format\": \"my_torch\"}\n"},
    {"empty.npy", ""},
    {"binary.npy", std::string("\x00\xff\x93NUMPY\x00", 9)},
};

Test(zip, crc32_matches_the_standard_check_value)
{
    cr_assert_eq(my_torch::crc32("123456789"), 0xCBF43926u);
    cr_assert_eq(my_torch::crc32(""), 0u);
}

Test(zip, reads_back_every_file_in_order)
{
    ZipEntries read = my_torch::from_zip(my_torch::to_zip(ENTRIES));

    cr_assert(read == ENTRIES);
}

Test(zip, the_same_files_always_give_the_same_bytes)
{
    cr_assert(my_torch::to_zip(ENTRIES) == my_torch::to_zip(ENTRIES));
}

Test(zip, starts_with_a_local_header_and_stores_files_as_they_are)
{
    std::string bytes = my_torch::to_zip(ENTRIES);

    cr_assert_eq(bytes.compare(0, 4, "PK\x03\x04"), 0);
    cr_assert_neq(bytes.find("{\"format\": \"my_torch\"}"), std::string::npos);
}

Test(zip, an_empty_archive_is_valid)
{
    cr_assert(my_torch::from_zip(my_torch::to_zip({})).empty());
}

Test(zip, refuses_a_damaged_file)
{
    std::string bytes = my_torch::to_zip(ENTRIES);

    bytes[bytes.find("my_torch")] = 'M';
    cr_assert_throw(my_torch::from_zip(bytes), ZipError);
}

Test(zip, refuses_a_truncated_archive_or_something_else)
{
    std::string bytes = my_torch::to_zip(ENTRIES);

    cr_assert_throw(my_torch::from_zip(bytes.substr(0, bytes.size() - 1)), ZipError);
    cr_assert_throw(my_torch::from_zip(bytes.substr(0, 40)), ZipError);
    cr_assert_throw(my_torch::from_zip(""), ZipError);
    cr_assert_throw(my_torch::from_zip(std::string(100, 'x')), ZipError);
}

Test(zip, refuses_compressed_files)
{
    std::string bytes = my_torch::to_zip({{"a", "b"}});
    std::size_t directory = bytes.find("PK\x01\x02");

    bytes[directory + 10] = 8;
    cr_assert_throw(my_torch::from_zip(bytes), ZipError);
}

Test(zip, refuses_a_damaged_directory)
{
    std::string bytes = my_torch::to_zip({{"a", "b"}});
    std::string local = bytes;

    bytes[bytes.find("PK\x01\x02") + 2] = 'x';
    local[2] = 'x';
    cr_assert_throw(my_torch::from_zip(bytes), ZipError);
    cr_assert_throw(my_torch::from_zip(local), ZipError);
}
