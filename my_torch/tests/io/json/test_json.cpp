/*
** EPITECH PROJECT, 2026
** my_torch
** File description:
** Tests of the JSON reader and writer
*/

#include <cmath>
#include <criterion/criterion.h>

#include "io/json/json.hpp"

using my_torch::Json;
using my_torch::JsonError;

static void assert_dump(const Json &json, const std::string &expected)
{
    std::string dumped = json.dump();

    cr_assert_str_eq(dumped.c_str(), expected.c_str());
}

Test(json, writes_every_kind_of_value)
{
    assert_dump(Json(), "null");
    assert_dump(Json(true), "true");
    assert_dump(Json(769), "769");
    assert_dump(Json(0.01), "0.01");
    assert_dump(Json(1e-12), "1e-12");
    assert_dump(Json("relu"), "\"relu\"");
    assert_dump(Json(Json::Array {1, "a", Json()}), "[1, \"a\", null]");
    assert_dump(Json(Json::Object {{"b", 1}, {"a", 2}}), "{\"b\": 1, \"a\": 2}");
}

Test(json, numbers_read_back_bit_for_bit)
{
    for (double number : {0.1, 1.0 / 3.0, -2.5e-300, 6.02214076e23, 4.9e-324, -0.0}) {
        double read = Json::parse(Json(number).dump()).as_number();
        cr_assert_eq(std::signbit(read), std::signbit(number));
        cr_assert_eq(read, number);
    }
}

Test(json, writes_nan_and_infinities_like_python)
{
    assert_dump(Json(NAN), "NaN");
    assert_dump(Json(INFINITY), "Infinity");
    assert_dump(Json(-INFINITY), "-Infinity");
    cr_assert(std::isnan(Json::parse("NaN").as_number()));
    cr_assert_eq(Json::parse("[Infinity, -Infinity]").as_array()[1].as_number(), -INFINITY);
}

Test(json, escapes_strings_both_ways)
{
    Json text("quote \" backslash \\ line\nend\ttab \x01");

    assert_dump(text, "\"quote \\\" backslash \\\\ line\\nend\\ttab \\u0001\"");
    cr_assert(Json::parse(text.dump()) == text);
    cr_assert(Json::parse("\"\\u00e9\\/\\b\\f\\r\"").as_string() == "\xc3\xa9/\b\f\r");
    cr_assert(Json::parse("\"\\ud83d\\ude00\"").as_string() == "\xf0\x9f\x98\x80");
}

Test(json, reads_a_nested_document)
{
    Json json = Json::parse(" {\"format\": \"my_torch\", \"layers\": [{\"shape\": [769, 256]}], \"ok\": false, \"none\": null}\n");

    cr_assert_str_eq(json.find("format")->as_string().c_str(), "my_torch");
    cr_assert_eq(json.find("layers")->as_array()[0].find("shape")->as_array()[1].as_number(), 256);
    cr_assert_not(json.find("ok")->as_bool());
    cr_assert(json.find("none")->is_null());
    cr_assert_null(json.find("missing"));
    cr_assert(Json::parse(json.dump()) == json);
}

Test(json, keeps_the_order_of_the_keys_and_replaces_with_set)
{
    Json json = Json::Object {{"z", 1}, {"a", 2}};

    json.set("a", 3);
    json.set("m", 4);
    assert_dump(json, "{\"z\": 1, \"a\": 3, \"m\": 4}");
}

Test(json, reads_empty_containers)
{
    cr_assert(Json::parse("[ ]").as_array().empty());
    cr_assert(Json::parse("{ }").as_object().empty());
}

Test(json, asking_for_the_wrong_type_throws)
{
    cr_assert_throw(Json(1).as_string(), JsonError);
    cr_assert_throw(Json("a").as_number(), JsonError);
    cr_assert_throw(Json().as_array(), JsonError);
    cr_assert_throw(Json(Json::Array {}).find("a"), JsonError);
    cr_assert_throw(Json(true).set("a", 1), JsonError);
}

Test(json, rejects_invalid_documents)
{
    for (const char *text : {"", "{", "[1, 2", "{\"a\" 1}", "{\"a\": 1,}", "\"open", "\"bad \\q\"", "\"\\u12\"",
             "\"\\uzzzz\"", "\"\\ud83d\\u0041\"", "nul", "1 2", "{\"a\": 1, \"a\": 2}", "1e", "-", "\"\\"})
        cr_assert_throw(Json::parse(text), JsonError, "%s should be refused", text);
}
