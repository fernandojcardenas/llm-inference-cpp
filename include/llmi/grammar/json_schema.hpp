#pragma once

#include <cstddef>
#include <string>

#include "llmi/util/json.hpp"
#include "llmi/util/result.hpp"

namespace llmi::grammar {

// Constrained decoding, part 3: JSON output (M8, ADR 0009).
//
// json_object_grammar() is any JSON object (OpenAI's response_format
// {"type": "json_object"}).
//
// json_schema_to_gbnf() turns a JSON Schema into grammar text for
// Grammar::parse(). It follows the rules of OpenAI's "strict" structured
// outputs, which keep the output fully determined by the schema:
//   - every object lists all of its properties in "required" and sets
//     "additionalProperties": false; properties are emitted in the order the
//     schema lists them;
//   - supported: type (one or a list), properties, required,
//     additionalProperties, items, minItems, maxItems, minLength, maxLength,
//     enum, const, anyOf, and the annotations title, description, default,
//     examples, $comment, $schema;
//   - any other keyword (pattern, format, minimum, $ref, ...) is an error, not
//     silently ignored: a grammar that ignored a constraint would let the
//     model produce output that violates the schema it was given.
// Whitespace between tokens is allowed but bounded (at most 20 characters at
// each spot), so a model cannot fill its whole token budget with spaces.

struct SchemaLimits {
  std::size_t max_depth = 16;
  std::size_t max_properties = 256;   // per object
  std::size_t max_enum_values = 256;
  std::size_t max_items = 1000;       // a larger maxItems is treated as unbounded
  std::size_t max_length = 1000;      // a larger maxLength is treated as unbounded
};

std::string json_object_grammar();

Result<std::string> json_schema_to_gbnf(const json::Value& schema, const SchemaLimits& limits = {});

}  // namespace llmi::grammar
