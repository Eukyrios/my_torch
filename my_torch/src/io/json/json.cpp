/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** JSON reading and writing. Numbers are written with the fewest digits that read back as the
** exact same double (0.01, not 0.01000000000000000021); NaN and infinities as Python writes them
*/

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <sstream>

#include "io/json/json.hpp"

namespace my_torch {

Json::Json(bool value) : _value(value) {}
Json::Json(int value) : _value(static_cast<double>(value)) {}
Json::Json(std::size_t value) : _value(static_cast<double>(value)) {}
Json::Json(double value) : _value(value) {}
Json::Json(const char *value) : _value(std::string(value)) {}
Json::Json(const std::string &value) : _value(value) {}
Json::Json(const Array &value) : _value(value) {}
Json::Json(const Object &value) : _value(value) {}

bool Json::is_null() const { return std::holds_alternative<std::nullptr_t>(_value); }
bool Json::is_bool() const { return std::holds_alternative<bool>(_value); }
bool Json::is_number() const { return std::holds_alternative<double>(_value); }
bool Json::is_string() const { return std::holds_alternative<std::string>(_value); }
bool Json::is_array() const { return std::holds_alternative<Array>(_value); }
bool Json::is_object() const { return std::holds_alternative<Object>(_value); }

template <typename Type>
static const Type &get(const auto &value, const char *expected)
{
    if (!std::holds_alternative<Type>(value))
        throw JsonError(std::string("expected ") + expected);
    return std::get<Type>(value);
}

bool Json::as_bool() const { return get<bool>(_value, "a boolean"); }
double Json::as_number() const { return get<double>(_value, "a number"); }
const std::string &Json::as_string() const { return get<std::string>(_value, "a string"); }
const Json::Array &Json::as_array() const { return get<Array>(_value, "an array"); }
const Json::Object &Json::as_object() const { return get<Object>(_value, "an object"); }

const Json *Json::find(const std::string &key) const
{
    for (const std::pair<std::string, Json> &member : as_object())
        if (member.first == key)
            return &member.second;
    return nullptr;
}

void Json::set(const std::string &key, const Json &value)
{
    if (!is_object())
        throw JsonError("expected an object");
    Object &members = std::get<Object>(_value);
    for (std::pair<std::string, Json> &member : members)
        if (member.first == key) {
            member.second = value;
            return;
        }
    members.emplace_back(key, value);
}

/* Writing */

static std::string dump_number(double number)
{
    if (std::isnan(number))
        return "NaN";
    if (std::isinf(number))
        return number > 0 ? "Infinity" : "-Infinity";
    char text[32];
    return std::string(text, std::to_chars(text, text + sizeof(text), number).ptr);
}

static std::string dump_string(const std::string &string)
{
    std::ostringstream text;
    text << '"';
    for (unsigned char character : string) {
        if (character == '"' || character == '\\')
            text << '\\' << character;
        else if (character == '\n')
            text << "\\n";
        else if (character == '\t')
            text << "\\t";
        else if (character < 0x20)
            text << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(character) << std::dec;
        else
            text << character;
    }
    text << '"';
    return text.str();
}

std::string Json::dump() const
{
    if (is_null())
        return "null";
    if (is_bool())
        return as_bool() ? "true" : "false";
    if (is_number())
        return dump_number(as_number());
    if (is_string())
        return dump_string(as_string());
    std::string text = is_array() ? "[" : "{";
    if (is_array())
        for (const Json &item : as_array())
            text += (text.size() > 1 ? ", " : "") + item.dump();
    else
        for (const std::pair<std::string, Json> &member : as_object())
            text += (text.size() > 1 ? ", " : "") + dump_string(member.first) + ": " + member.second.dump();
    return text + (is_array() ? "]" : "}");
}

/* Reading */

namespace {

class Parser {
public:
    explicit Parser(const std::string &text) : _text(text) {}

    Json document()
    {
        Json value = this->value();
        skip_spaces();
        if (_position != _text.size())
            fail("unexpected text after the value");
        return value;
    }

private:
    [[noreturn]] void fail(const std::string &message) const
    {
        throw JsonError("invalid JSON at character " + std::to_string(_position) + ": " + message);
    }

    void skip_spaces()
    {
        while (_position < _text.size() && std::string(" \t\n\r").find(_text[_position]) != std::string::npos)
            _position++;
    }

    bool next_is(const std::string &word)
    {
        if (_text.compare(_position, word.size(), word) != 0)
            return false;
        _position += word.size();
        return true;
    }

    void expect(char character)
    {
        skip_spaces();
        if (_position >= _text.size() || _text[_position] != character)
            fail(std::string("expected '") + character + "'");
        _position++;
    }

    Json value()
    {
        skip_spaces();
        if (_position >= _text.size())
            fail("unexpected end");
        char first = _text[_position];
        if (first == '{')
            return object();
        if (first == '[')
            return array();
        if (first == '"')
            return Json(string());
        if (next_is("null"))
            return Json();
        if (next_is("true"))
            return Json(true);
        if (next_is("false"))
            return Json(false);
        return Json(number());
    }

    double number()
    {
        if (next_is("NaN"))
            return NAN;
        if (next_is("Infinity"))
            return INFINITY;
        if (next_is("-Infinity"))
            return -INFINITY;
        std::size_t start = _position;
        while (_position < _text.size() && std::string("+-0123456789.eE").find(_text[_position]) != std::string::npos)
            _position++;
        std::string digits = _text.substr(start, _position - start);
        char *end = nullptr;
        double number = digits.empty() ? 0.0 : std::strtod(digits.c_str(), &end);
        if (digits.empty() || end != digits.c_str() + digits.size())
            fail("expected a value");
        return number;
    }

    unsigned hex_code()
    {
        if (_position + 4 > _text.size())
            fail("unfinished \\u escape");
        std::string hex = _text.substr(_position, 4);
        char *end = nullptr;
        unsigned code = std::strtoul(hex.c_str(), &end, 16);
        if (end != hex.c_str() + 4)
            fail("invalid \\u escape");
        _position += 4;
        return code;
    }

    static void append_utf8(std::string &text, std::uint32_t code)
    {
        if (code < 0x80) {
            text += static_cast<char>(code);
        } else if (code < 0x800) {
            text += static_cast<char>(0xC0 | (code >> 6));
            text += static_cast<char>(0x80 | (code & 0x3F));
        } else if (code < 0x10000) {
            text += static_cast<char>(0xE0 | (code >> 12));
            text += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            text += static_cast<char>(0x80 | (code & 0x3F));
        } else {
            text += static_cast<char>(0xF0 | (code >> 18));
            text += static_cast<char>(0x80 | ((code >> 12) & 0x3F));
            text += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
            text += static_cast<char>(0x80 | (code & 0x3F));
        }
    }

    std::uint32_t unicode_escape()
    {
        std::uint32_t code = hex_code();
        // a character above U+FFFF comes as two escapes, a surrogate pair
        if (code >= 0xD800 && code < 0xDC00 && next_is("\\u")) {
            std::uint32_t low = hex_code();
            if (low < 0xDC00 || low >= 0xE000)
                fail("invalid surrogate pair");
            code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
        }
        return code;
    }

    std::string string()
    {
        expect('"');
        std::string text;
        while (true) {
            if (_position >= _text.size())
                fail("unfinished string");
            char character = _text[_position++];
            if (character == '"')
                return text;
            if (character != '\\') {
                text += character;
                continue;
            }
            if (_position >= _text.size())
                fail("unfinished string");
            char escaped = _text[_position++];
            std::string from = "\"\\/bfnrt";
            std::string to = "\"\\/\b\f\n\r\t";
            if (escaped == 'u')
                append_utf8(text, unicode_escape());
            else if (from.find(escaped) != std::string::npos)
                text += to[from.find(escaped)];
            else
                fail(std::string("invalid escape \\") + escaped);
        }
    }

    Json array()
    {
        expect('[');
        Json::Array items;
        skip_spaces();
        if (next_is("]"))
            return Json(items);
        do {
            items.push_back(value());
            skip_spaces();
        } while (next_is(","));
        expect(']');
        return Json(items);
    }

    Json object()
    {
        expect('{');
        Json members = Json(Json::Object());
        skip_spaces();
        if (next_is("}"))
            return members;
        do {
            skip_spaces();
            std::string key = string();
            if (members.find(key))
                fail("key '" + key + "' given twice");
            expect(':');
            members.set(key, value());
            skip_spaces();
        } while (next_is(","));
        expect('}');
        return members;
    }

    const std::string &_text;
    std::size_t _position = 0;
};

}

Json Json::parse(const std::string &text)
{
    return Parser(text).document();
}

}
