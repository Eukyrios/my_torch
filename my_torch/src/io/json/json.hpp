/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** A JSON value (null, bool, number, string, array or object), read with parse and written with
** dump; objects keep their keys in order, numbers are doubles written back without losing a bit
*/

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace my_torch {

class JsonError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class Json {
public:
    using Array = std::vector<Json>;
    using Object = std::vector<std::pair<std::string, Json>>;

    Json() = default;
    Json(bool value);
    Json(int value);
    Json(std::size_t value);
    Json(double value);
    Json(const char *value);
    Json(const std::string &value);
    Json(const Array &value);
    Json(const Object &value);

    static Json parse(const std::string &text);
    std::string dump() const;

    bool is_null() const;
    bool is_bool() const;
    bool is_number() const;
    bool is_string() const;
    bool is_array() const;
    bool is_object() const;
    bool as_bool() const;
    double as_number() const;
    const std::string &as_string() const;
    const Array &as_array() const;
    const Object &as_object() const;
    const Json *find(const std::string &key) const;
    void set(const std::string &key, const Json &value);

    bool operator==(const Json &other) const = default;

private:
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> _value = nullptr;
};

}
