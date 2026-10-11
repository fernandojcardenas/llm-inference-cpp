#include "llmi/grammar/json_schema.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <string_view>
#include <vector>

namespace llmi::grammar {

namespace {

// Shared building blocks. String length is counted in JSON characters: an
// escape such as \n or é counts as one.
constexpr std::string_view kPrimitives = R"(ws ::= [ \t\n]{0,20}
char ::= [^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4})
string ::= "\"" char* "\""
number ::= "-"? ("0" | [1-9] [0-9]{0,15}) ("." [0-9]{1,16})? ([eE] [-+]? [0-9]{1,3})?
integer ::= "-"? ("0" | [1-9] [0-9]{0,15})
boolean ::= "true" | "false"
null ::= "null"
)";

constexpr std::string_view kGenericValue = R"(value ::= object | array | string | number | boolean | null
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
)";

// A GBNF string literal whose characters are exactly `bytes`.
std::string literal(std::string_view bytes) {
  static constexpr std::array<char, 16> kHex = {'0', '1', '2', '3', '4', '5', '6', '7',
                                                '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
  std::string out = "\"";
  for (const char c : bytes) {
    const auto u = static_cast<unsigned char>(c);
    if (c == '\\' || c == '"') {
      out += '\\';
      out += c;
    } else if (u < 0x20) {
      out += "\\x";
      out += kHex.at(u >> 4U);
      out += kHex.at(u & 0x0FU);
    } else {
      out += c;
    }
  }
  out += '"';
  return out;
}

// Compact JSON text of a value: how an enum or const value must appear in
// the output.
// Recursion depth is bounded by the JSON parser's own nesting limit.
// NOLINTNEXTLINE(misc-no-recursion)
void serialize(const json::Value& v, std::string& out) {
  switch (v.kind()) {
    case json::Value::Kind::Null: out += "null"; break;
    case json::Value::Kind::Bool: out += v.as_bool() ? "true" : "false"; break;
    case json::Value::Kind::Number: out += v.text(); break;
    case json::Value::Kind::String: json::append_quoted(out, v.text()); break;
    case json::Value::Kind::Array: {
      out += '[';
      bool first = true;
      for (const json::Value& item : v.items()) {
        if (!first) out += ',';
        first = false;
        serialize(item, out);
      }
      out += ']';
      break;
    }
    case json::Value::Kind::Object: {
      out += '{';
      bool first = true;
      for (const auto& [key, value] : v.members()) {
        if (!first) out += ',';
        first = false;
        json::append_quoted(out, key);
        out += ':';
        serialize(value, out);
      }
      out += '}';
      break;
    }
  }
}

std::string rule_name_part(std::string_view s) {
  std::string out;
  for (const char c : s) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
    out += ok ? c : '-';
    if (out.size() >= 24) break;
  }
  return out.empty() ? "x" : out;
}

class Converter {
 public:
  explicit Converter(const SchemaLimits& limits) : limits_(limits) {}

  Result<std::string> run(const json::Value& schema) {
    std::string root_expr;
    if (!visit(schema, "root", 0, root_expr)) return fail(error_);
    std::string out = "root ::= " + root_expr + "\n";
    for (const std::string& r : rules_) out += r;
    out += kPrimitives;
    return out;
  }

 private:
  bool error(const std::string& where, const std::string& message) {
    error_ = "JSON schema at " + where + ": " + message;
    return false;
  }

  std::string add_rule(const std::string& hint, const std::string& body) {
    std::string name = rule_name_part(hint) + "-" + std::to_string(++counter_);
    rules_.push_back(name + " ::= " + body + "\n");
    return name;
  }

  static bool is_annotation(std::string_view key) {
    return key == "title" || key == "description" || key == "default" || key == "examples" || key == "$comment" ||
           key == "$schema";
  }

  bool check_keywords(const json::Value& schema, const std::string& where, std::initializer_list<std::string_view> ok) {
    for (const auto& [key, value] : schema.members()) {
      if (is_annotation(key)) continue;
      if (std::find(ok.begin(), ok.end(), key) == ok.end()) {
        return error(where, "keyword '" + key + "' is not supported (see ADR 0009)");
      }
    }
    return true;
  }

  bool count(const json::Value* v, const std::string& where, const char* name, std::size_t& out) {
    if (v == nullptr) return true;
    const auto n = v->as_u64();
    if (!n) return error(where, std::string(name) + " must be a non-negative integer");
    out = static_cast<std::size_t>(*n);
    return true;
  }

  // visit() and the visit_* functions recurse into sub-schemas; depth is
  // bounded by SchemaLimits::max_depth, checked first thing here.
  // NOLINTNEXTLINE(misc-no-recursion)
  bool visit(const json::Value& schema, const std::string& where, std::size_t depth, std::string& out) {
    if (depth > limits_.max_depth) return error(where, "nested deeper than " + std::to_string(limits_.max_depth));
    if (!schema.is_object()) return error(where, "a schema must be a JSON object");
    // Unknown keywords first, so "$ref" says "$ref is not supported" rather
    // than "needs a type".
    if (!check_keywords(schema, where,
                        {"type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
                         "minLength", "maxLength", "enum", "const", "anyOf"})) {
      return false;
    }

    if (const json::Value* c = schema.find("const")) {
      if (!check_keywords(schema, where, {"const", "type"})) return false;
      std::string text;
      serialize(*c, text);
      out = literal(text);
      return true;
    }
    if (const json::Value* e = schema.find("enum")) {
      if (!check_keywords(schema, where, {"enum", "type"})) return false;
      if (!e->is_array() || e->items().empty()) return error(where, "enum must be a non-empty array");
      if (e->items().size() > limits_.max_enum_values) return error(where, "enum has too many values");
      std::string alts;
      for (const json::Value& v : e->items()) {
        std::string text;
        serialize(v, text);
        if (!alts.empty()) alts += " | ";
        alts += literal(text);
      }
      out = add_rule(where, alts);
      return true;
    }
    if (const json::Value* any = schema.find("anyOf")) {
      if (!check_keywords(schema, where, {"anyOf"})) return false;
      if (!any->is_array() || any->items().empty()) return error(where, "anyOf must be a non-empty array");
      std::string alts;
      for (std::size_t i = 0; i < any->items().size(); ++i) {
        std::string sub;
        if (!visit(any->items()[i], where + "/anyOf/" + std::to_string(i), depth + 1, sub)) return false;
        if (!alts.empty()) alts += " | ";
        alts += sub;
      }
      out = add_rule(where, alts);
      return true;
    }

    const json::Value* type = schema.find("type");
    if (type == nullptr) return error(where, "needs a \"type\" (or enum, const, anyOf)");
    if (type->is_array()) {
      // "type": ["string", "null"] = anyOf over the listed types, each with
      // the rest of this schema's keywords.
      if (type->items().empty()) return error(where, "type list is empty");
      std::string alts;
      for (const json::Value& t : type->items()) {
        if (!t.is_string()) return error(where, "type list entries must be strings");
        std::string sub;
        if (!visit_type(schema, t.text(), where, depth, sub)) return false;
        if (!alts.empty()) alts += " | ";
        alts += sub;
      }
      out = add_rule(where, alts);
      return true;
    }
    if (!type->is_string()) return error(where, "type must be a string or a list of strings");
    return visit_type(schema, type->text(), where, depth, out);
  }

  // NOLINTNEXTLINE(misc-no-recursion): bounded by max_depth, see visit()
  bool visit_type(const json::Value& schema, const std::string& type, const std::string& where, std::size_t depth,
                  std::string& out) {
    if (type == "object") return visit_object(schema, where, depth, out);
    if (type == "array") return visit_array(schema, where, depth, out);
    if (type == "string") return visit_string(schema, where, out);
    if (type == "number" || type == "integer" || type == "boolean" || type == "null") {
      if (!check_keywords(schema, where, {"type"})) return false;
      out = type;
      return true;
    }
    return error(where, "unknown type '" + type + "'");
  }

  bool visit_string(const json::Value& schema, const std::string& where, std::string& out) {
    if (!check_keywords(schema, where, {"type", "minLength", "maxLength"})) return false;
    std::size_t min = 0;
    std::size_t max = 0;
    const json::Value* max_v = schema.find("maxLength");
    if (!count(schema.find("minLength"), where, "minLength", min)) return false;
    if (!count(max_v, where, "maxLength", max)) return false;
    if (min == 0 && max_v == nullptr) {
      out = "string";
      return true;
    }
    if (min > limits_.max_length) return error(where, "minLength over " + std::to_string(limits_.max_length));
    std::string rep;
    if (max_v == nullptr || max > limits_.max_length) {
      rep = "{" + std::to_string(min) + ",}";
    } else {
      if (max < min) return error(where, "maxLength is less than minLength");
      rep = "{" + std::to_string(min) + "," + std::to_string(max) + "}";
    }
    out = add_rule(where, R"("\"" char)" + rep + R"( "\"")");
    return true;
  }

  // NOLINTNEXTLINE(misc-no-recursion): bounded by max_depth, see visit()
  bool visit_array(const json::Value& schema, const std::string& where, std::size_t depth, std::string& out) {
    if (!check_keywords(schema, where, {"type", "items", "minItems", "maxItems"})) return false;
    const json::Value* items = schema.find("items");
    if (items == nullptr) return error(where, "arrays need an \"items\" schema");
    std::string item;
    if (!visit(*items, where + "/items", depth + 1, item)) return false;
    std::size_t min = 0;
    std::size_t max = 0;
    const json::Value* max_v = schema.find("maxItems");
    if (!count(schema.find("minItems"), where, "minItems", min)) return false;
    if (!count(max_v, where, "maxItems", max)) return false;
    const bool unbounded = max_v == nullptr || max > limits_.max_items;
    if (min > limits_.max_items) return error(where, "minItems over " + std::to_string(limits_.max_items));
    if (!unbounded && max < min) return error(where, "maxItems is less than minItems");
    if (!unbounded && max == 0) {
      out = add_rule(where, R"("[" ws "]")");
      return true;
    }
    const std::string more = "(ws \",\" ws " + item + ")";
    const std::size_t extra_min = min == 0 ? 0 : min - 1;
    const std::string rep = unbounded ? "{" + std::to_string(extra_min) + ",}"
                                      : "{" + std::to_string(extra_min) + "," + std::to_string(max - 1) + "}";
    std::string list = item + " " + more + rep;
    if (min == 0) list = "(" + list + ")?";
    out = add_rule(where, "\"[\" ws " + list + " ws \"]\"");
    return true;
  }

  // NOLINTNEXTLINE(misc-no-recursion): bounded by max_depth, see visit()
  bool visit_object(const json::Value& schema, const std::string& where, std::size_t depth, std::string& out) {
    if (!check_keywords(schema, where, {"type", "properties", "required", "additionalProperties"})) return false;
    const json::Value* ap = schema.find("additionalProperties");
    if (ap == nullptr || !ap->is_bool() || ap->as_bool()) {
      return error(where, "objects must set \"additionalProperties\": false (strict mode)");
    }
    const json::Value* props = schema.find("properties");
    if (props != nullptr && !props->is_object()) return error(where, "properties must be an object");
    const std::vector<json::Value::Member> empty;
    const auto& members = props != nullptr ? props->members() : empty;
    if (members.size() > limits_.max_properties) return error(where, "too many properties");

    // Strict mode: "required" lists exactly the properties.
    std::set<std::string> required;
    if (const json::Value* req = schema.find("required")) {
      if (!req->is_array()) return error(where, "required must be an array");
      for (const json::Value& r : req->items()) {
        if (!r.is_string()) return error(where, "required entries must be strings");
        required.insert(r.text());
      }
    }
    std::set<std::string> names;
    for (const auto& [key, value] : members) names.insert(key);
    if (required != names) {
      return error(where, "every property must be listed in \"required\", and nothing else (strict mode)");
    }

    std::string body = R"("{" ws )";
    bool first = true;
    for (const auto& [key, value] : members) {
      std::string sub;
      std::string path = where;
      path += '/';
      path += key;
      if (!visit(value, path, depth + 1, sub)) return false;
      std::string quoted;
      json::append_quoted(quoted, key);
      if (!first) body += R"("," ws )";
      first = false;
      body += literal(quoted);
      body += R"( ws ":" ws )";
      body += sub;
      body += " ws ";
    }
    body += R"("}")";
    out = add_rule(where, body);
    return true;
  }

  const SchemaLimits& limits_;
  std::vector<std::string> rules_;
  std::size_t counter_ = 0;
  std::string error_;
};

}  // namespace

std::string json_object_grammar() {
  std::string out = "root ::= object\n";
  out += kGenericValue;
  out += kPrimitives;
  return out;
}

Result<std::string> json_schema_to_gbnf(const json::Value& schema, const SchemaLimits& limits) {
  return Converter(limits).run(schema);
}

}  // namespace llmi::grammar
