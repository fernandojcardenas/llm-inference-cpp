#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "llmi/grammar/grammar.hpp"
#include "llmi/grammar/json_schema.hpp"
#include "llmi/grammar/matcher.hpp"
#include "llmi/util/json.hpp"

using llmi::grammar::Grammar;
using llmi::grammar::Matcher;

namespace {

std::shared_ptr<const Grammar> from_schema(const std::string& schema_json) {
  auto schema = llmi::json::parse(schema_json);
  EXPECT_TRUE(schema.ok());
  if (!schema.ok()) return nullptr;
  auto text = llmi::grammar::json_schema_to_gbnf(*schema);
  EXPECT_TRUE(text.ok()) << (text.ok() ? "" : text.error());
  if (!text.ok()) return nullptr;
  auto g = Grammar::parse(*text);
  EXPECT_TRUE(g.ok()) << (g.ok() ? "" : g.error()) << "\n" << *text;
  if (!g.ok()) return nullptr;
  return std::make_shared<const Grammar>(std::move(g.value()));
}

std::string schema_error(const std::string& schema_json) {
  auto schema = llmi::json::parse(schema_json);
  if (!schema.ok()) return "bad test JSON";
  auto text = llmi::grammar::json_schema_to_gbnf(*schema);
  return text.ok() ? std::string("(converted)") : text.error();
}

bool matches(const std::shared_ptr<const Grammar>& g, const std::string& s) {
  if (!g) return false;
  Matcher m(g);
  return m.accept(s) && m.complete();
}

constexpr const char* kPerson = R"({
  "type": "object",
  "properties": {
    "name": {"type": "string", "minLength": 1, "maxLength": 20},
    "age": {"type": "integer"},
    "role": {"enum": ["admin", "user", null]},
    "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
    "score": {"type": ["number", "null"]},
    "ok": {"type": "boolean", "description": "annotations are ignored"}
  },
  "required": ["name", "age", "role", "tags", "score", "ok"],
  "additionalProperties": false
})";

}  // namespace

TEST(JsonSchema, ObjectWithEveryPropertyInSchemaOrder) {
  auto g = from_schema(kPerson);
  EXPECT_TRUE(matches(g, R"({"name":"Ada","age":36,"role":"admin","tags":["x","y"],"score":0.5,"ok":true})"));
  EXPECT_TRUE(matches(g, "{ \"name\": \"B\",\n \"age\": -3, \"role\": null, \"tags\": [], \"score\": null, \"ok\": false }"));
  EXPECT_TRUE(matches(g, R"({"name":"é\n","age":0,"role":"user","tags":["a","b","c"],"score":1e10,"ok":true})"));
  // Missing, reordered or extra properties; wrong types; constraint breaks.
  EXPECT_FALSE(matches(g, R"({"name":"Ada","age":36})"));
  EXPECT_FALSE(matches(g, R"({"age":36,"name":"Ada","role":"admin","tags":[],"score":1,"ok":true})"));
  EXPECT_FALSE(matches(g, R"({"name":"Ada","age":3.5,"role":"admin","tags":[],"score":1,"ok":true})"));
  EXPECT_FALSE(matches(g, R"({"name":"","age":1,"role":"admin","tags":[],"score":1,"ok":true})"));
  EXPECT_FALSE(matches(g, R"({"name":"A","age":1,"role":"root","tags":[],"score":1,"ok":true})"));
  EXPECT_FALSE(matches(g, R"({"name":"A","age":1,"role":"user","tags":["1","2","3","4"],"score":1,"ok":true})"));
  EXPECT_FALSE(matches(g, R"({"name":"A","age":01,"role":"user","tags":[],"score":1,"ok":true})"));
  EXPECT_FALSE(matches(g, R"({"name":"A","age":1,"role":"user","tags":[],"score":1,"ok":true,"x":1})"));
}

TEST(JsonSchema, WhitespaceIsBounded) {
  auto g = from_schema(R"({"type":"object","properties":{},"required":[],"additionalProperties":false})");
  EXPECT_TRUE(matches(g, "{" + std::string(20, ' ') + "}"));
  EXPECT_FALSE(matches(g, "{" + std::string(41, ' ') + "}"));  // two ws spots of at most 20 each
}

TEST(JsonSchema, ArraysConstAnyOfAndNesting) {
  auto g = from_schema(R"({
    "type": "array", "minItems": 2, "maxItems": 3,
    "items": {"anyOf": [
      {"const": {"k": [1, "two"]}},
      {"type": "object", "properties": {"inner": {"type": "array", "items": {"type": "integer"}}},
       "required": ["inner"], "additionalProperties": false}
    ]}
  })");
  EXPECT_TRUE(matches(g, R"([{"k":[1,"two"]},{"inner":[1,2,3]}])"));
  EXPECT_TRUE(matches(g, R"([{"inner":[]}, {"inner":[0]}, {"k":[1,"two"]}])"));
  EXPECT_FALSE(matches(g, R"([{"inner":[]}])"));                                  // minItems 2
  EXPECT_FALSE(matches(g, R"([{"inner":[]},{"inner":[]},{"inner":[]},{"inner":[]}])"));  // maxItems 3
  EXPECT_FALSE(matches(g, R"([{"k":[1,2]},{"inner":[]}])"));                      // const differs
}

TEST(JsonSchema, StrictModeAndUnsupportedKeywordsAreErrors) {
  EXPECT_NE(schema_error(R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]})")
                .find("additionalProperties"),
            std::string::npos);
  EXPECT_NE(schema_error(R"({"type":"object","properties":{"a":{"type":"string"}},"required":[],
                             "additionalProperties":false})")
                .find("required"),
            std::string::npos);
  EXPECT_NE(schema_error(R"({"type":"string","pattern":"^a+$"})").find("'pattern' is not supported"), std::string::npos);
  EXPECT_NE(schema_error(R"({"type":"integer","minimum":0})").find("'minimum' is not supported"), std::string::npos);
  EXPECT_NE(schema_error(R"({"$ref":"#/$defs/x"})").find("'$ref' is not supported"), std::string::npos);
  EXPECT_NE(schema_error(R"({"type":"date"})").find("unknown type"), std::string::npos);
  EXPECT_NE(schema_error(R"({"description":"no type"})").find("needs a \"type\""), std::string::npos);
  EXPECT_NE(schema_error(R"({"type":"array"})").find("\"items\""), std::string::npos);
  EXPECT_NE(schema_error(R"({"enum":[]})").find("non-empty"), std::string::npos);
  EXPECT_NE(schema_error(R"({"type":"string","minLength":5,"maxLength":2})").find("less than"), std::string::npos);
  std::string deep = R"({"type":"string"})";
  for (int i = 0; i < 20; ++i) deep = R"({"type":"array","items":)" + deep + "}";
  EXPECT_NE(schema_error(deep).find("nested deeper"), std::string::npos);
}

TEST(JsonSchema, PropertyNamesAndEnumStringsAreEscapedExactly) {
  auto g = from_schema(R"({"type":"object","properties":{"a \"q\" \\ b":{"enum":["x\ny", "é"]}},
                           "required":["a \"q\" \\ b"],"additionalProperties":false})");
  EXPECT_TRUE(matches(g, R"({"a \"q\" \\ b":"x\ny"})"));
  EXPECT_TRUE(matches(g, "{\"a \\\"q\\\" \\\\ b\":\"\xc3\xa9\"}"));
  EXPECT_FALSE(matches(g, R"({"a \"q\" \\ b":"x y"})"));
}

TEST(JsonSchema, AnyJsonObject) {
  auto g = Grammar::parse(llmi::grammar::json_object_grammar());
  ASSERT_TRUE(g.ok()) << g.error();
  auto shared = std::make_shared<const Grammar>(std::move(g.value()));
  EXPECT_TRUE(matches(shared, R"({"a":[1,2,{"b":null}],"c":"d","e":-1.5E3,"f":true})"));
  EXPECT_TRUE(matches(shared, "{}"));
  EXPECT_FALSE(matches(shared, "[]"));   // the top level must be an object
  EXPECT_FALSE(matches(shared, R"({"a":})"));
  EXPECT_FALSE(matches(shared, R"({"a":1,})"));
}
