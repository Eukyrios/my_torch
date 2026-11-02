/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A zip is a local header and the data of each file, then a central directory listing them
** and an end record giving where that directory starts. Files are stored as they are (method 0)
** and dated 1980-01-01, so the same entries always give the same bytes; no zip64, so each file
** and the whole archive stay under 4 GB
*/

#include <array>
#include <limits>

#include "io/zip/zip.hpp"

namespace my_torch {

static const std::uint32_t LOCAL_SIGNATURE = 0x04034b50;
static const std::uint32_t CENTRAL_SIGNATURE = 0x02014b50;
static const std::uint32_t END_SIGNATURE = 0x06054b50;
static const std::size_t LOCAL_SIZE = 30;
static const std::size_t CENTRAL_SIZE = 46;
static const std::size_t END_SIZE = 22;
static const std::uint16_t VERSION = 20;
static const std::uint16_t STORED = 0;
static const std::uint16_t DOS_DATE_1980 = (0 << 9) | (1 << 5) | 1;

static std::array<std::uint32_t, 256> crc_table()
{
    std::array<std::uint32_t, 256> table {};
    for (std::uint32_t byte = 0; byte < 256; byte++) {
        std::uint32_t crc = byte;
        for (int bit = 0; bit < 8; bit++)
            crc = crc & 1 ? 0xEDB88320 ^ (crc >> 1) : crc >> 1;
        table[byte] = crc;
    }
    return table;
}

std::uint32_t crc32(const std::string &bytes)
{
    static const std::array<std::uint32_t, 256> table = crc_table();
    std::uint32_t crc = 0xFFFFFFFF;
    for (unsigned char byte : bytes)
        crc = table[(crc ^ byte) & 0xFF] ^ (crc >> 8);
    return crc ^ 0xFFFFFFFF;
}

static void put(std::string &bytes, std::uint32_t value, std::size_t size)
{
    for (std::size_t index = 0; index < size; index++)
        bytes += static_cast<char>((value >> (8 * index)) & 0xFF);
}

static void check_fits(std::size_t size, const std::string &what)
{
    if (size > std::numeric_limits<std::uint32_t>::max())
        throw ZipError(what + " is too big for a zip without zip64");
}

// the fields shared by a local header and a central directory entry, from the version needed
static void put_file_fields(std::string &bytes, const std::string &name, const std::string &data)
{
    put(bytes, VERSION, 2);
    put(bytes, 0, 2);
    put(bytes, STORED, 2);
    put(bytes, 0, 2);
    put(bytes, DOS_DATE_1980, 2);
    put(bytes, crc32(data), 4);
    put(bytes, data.size(), 4);
    put(bytes, data.size(), 4);
    put(bytes, name.size(), 2);
    put(bytes, 0, 2);
}

std::string to_zip(const ZipEntries &entries)
{
    if (entries.size() > std::numeric_limits<std::uint16_t>::max())
        throw ZipError("too many files for a zip without zip64");
    std::string bytes;
    std::string directory;
    for (const auto &[name, data] : entries) {
        check_fits(data.size(), "'" + name + "'");
        std::size_t offset = bytes.size();
        put(bytes, LOCAL_SIGNATURE, 4);
        put_file_fields(bytes, name, data);
        bytes += name + data;
        put(directory, CENTRAL_SIGNATURE, 4);
        put(directory, VERSION, 2);
        put_file_fields(directory, name, data);
        put(directory, 0, 2);
        put(directory, 0, 2);
        put(directory, 0, 2);
        put(directory, 0, 4);
        put(directory, offset, 4);
        directory += name;
    }
    check_fits(bytes.size() + directory.size(), "the archive");
    std::size_t directory_offset = bytes.size();
    bytes += directory;
    put(bytes, END_SIGNATURE, 4);
    put(bytes, 0, 2);
    put(bytes, 0, 2);
    put(bytes, entries.size(), 2);
    put(bytes, entries.size(), 2);
    put(bytes, directory.size(), 4);
    put(bytes, directory_offset, 4);
    put(bytes, 0, 2);
    return bytes;
}

/* Reading */

static std::uint32_t get(const std::string &bytes, std::size_t offset, std::size_t size)
{
    if (offset + size > bytes.size())
        throw ZipError("truncated archive");
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < size; index++)
        value |= static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[offset + index])) << (8 * index);
    return value;
}

static std::string slice(const std::string &bytes, std::size_t offset, std::size_t size)
{
    if (offset + size > bytes.size())
        throw ZipError("truncated archive");
    return bytes.substr(offset, size);
}

// the end record is the last thing in the file, followed only by a comment of up to 65535 bytes
static std::size_t find_end_record(const std::string &bytes)
{
    if (bytes.size() < END_SIZE)
        throw ZipError("not a zip archive");
    std::size_t lowest = bytes.size() > END_SIZE + 0xFFFF ? bytes.size() - END_SIZE - 0xFFFF : 0;
    for (std::size_t offset = bytes.size() - END_SIZE + 1; offset-- > lowest;)
        if (get(bytes, offset, 4) == END_SIGNATURE)
            return offset;
    throw ZipError("not a zip archive");
}

static std::pair<std::string, std::string> read_entry(const std::string &bytes, std::size_t &position)
{
    if (get(bytes, position, 4) != CENTRAL_SIGNATURE)
        throw ZipError("damaged central directory");
    std::uint16_t method = get(bytes, position + 10, 2);
    std::uint32_t crc = get(bytes, position + 16, 4);
    std::uint32_t size = get(bytes, position + 20, 4);
    std::size_t name_size = get(bytes, position + 28, 2);
    std::size_t extra_size = get(bytes, position + 30, 2);
    std::size_t comment_size = get(bytes, position + 32, 2);
    std::size_t offset = get(bytes, position + 42, 4);
    std::string name = slice(bytes, position + CENTRAL_SIZE, name_size);
    position += CENTRAL_SIZE + name_size + extra_size + comment_size;
    if (method != STORED)
        throw ZipError("'" + name + "' is compressed, only stored files are supported");
    if (get(bytes, offset, 4) != LOCAL_SIGNATURE)
        throw ZipError("damaged header for '" + name + "'");
    std::size_t data_offset = offset + LOCAL_SIZE + get(bytes, offset + 26, 2) + get(bytes, offset + 28, 2);
    std::string data = slice(bytes, data_offset, size);
    if (crc32(data) != crc)
        throw ZipError("'" + name + "' is damaged (wrong CRC-32)");
    return {name, data};
}

ZipEntries from_zip(const std::string &bytes)
{
    std::size_t end = find_end_record(bytes);
    std::size_t count = get(bytes, end + 10, 2);
    std::size_t position = get(bytes, end + 16, 4);
    ZipEntries entries;
    for (std::size_t index = 0; index < count; index++)
        entries.push_back(read_entry(bytes, position));
    return entries;
}

}
