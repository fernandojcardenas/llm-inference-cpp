#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llmi/util/result.hpp"

namespace llmi::json {

// A small JSON parser (RFC 8259) for the files this engine reads: the
// safetensors header, config.json and tokenizer.json. All three can come
// from untrusted sources, so the parser is strict and bounded:
//   - input must be valid UTF-8, and so must every decoded string;
//   - nesting depth and total node count are limited;
//   - duplicate keys in an object are an error;
//   - numbers keep their source text, so 64-bit integers are read exactly.
struct Limits {
  std::size_t max_depth = 64;
  std::size_t max_nodes = 50'000'000;
};

class Value {
 public:
  enum class Kind : std::uint8_t { Null, Bool, Number, String, Array, Object };
  using Member = std::pair<std::string, Value>;

  Value() = default;
  static Value make_bool(bool b);
  static Value make_number(std::string text);
  static Value make_string(std::string s);
  static Value make_array(std::vector<Value> items);
  static Value make_object(std::vector<Member> members);

  [[nodiscard]] Kind kind() const { return kind_; }
  [[nodiscard]] bool is_null() const { return kind_ == Kind::Null; }
  [[nodiscard]] bool is_bool() const { return kind_ == Kind::Bool; }
  [[nodiscard]] bool is_number() const { return kind_ == Kind::Number; }
  [[nodiscard]] bool is_string() const { return kind_ == Kind::String; }
  [[nodiscard]] bool is_array() const { return kind_ == Kind::Array; }
  [[nodiscard]] bool is_object() const { return kind_ == Kind::Object; }

  [[nodiscard]] bool as_bool() const { return bool_; }
  // The string value (String) or the number's source text (Number).
  [[nodiscard]] const std::string& text() const { return text_; }
  [[nodiscard]] const std::vector<Value>& items() const { return items_; }
  [[nodiscard]] const std::vector<Member>& members() const { return members_; }

  // Object member by key, or nullptr. Linear scan: objects here are small,
  // except tokenizer vocabularies, which are iterated, not searched.
  [[nodiscard]] const Value* find(std::string_view key) const;

  // Exact conversions; nullopt if the value is not a number of that kind.
  [[nodiscard]] std::optional<std::uint64_t> as_u64() const;
  [[nodiscard]] std::optional<std::int64_t> as_i64() const;
  [[nodiscard]] std::optional<double> as_double() const;

 private:
  Kind kind_ = Kind::Null;
  bool bool_ = false;
  std::string text_;
  std::vector<Value> items_;
  std::vector<Member> members_;
};

Result<Value> parse(std::string_view input, const Limits& limits = {});

// Appends `s` to `out` as a JSON string literal (with quotes).
void append_quoted(std::string& out, std::string_view s);

}  // namespace llmi::json
