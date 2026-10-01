#include "llmi/util/json.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <unordered_set>

#include "llmi/util/utf8.hpp"

namespace llmi::json {

Value Value::make_bool(bool b) {
  Value v;
  v.kind_ = Kind::Bool;
  v.bool_ = b;
  return v;
}
Value Value::make_number(std::string text) {
  Value v;
  v.kind_ = Kind::Number;
  v.text_ = std::move(text);
  return v;
}
Value Value::make_string(std::string s) {
  Value v;
  v.kind_ = Kind::String;
  v.text_ = std::move(s);
  return v;
}
Value Value::make_array(std::vector<Value> items) {
  Value v;
  v.kind_ = Kind::Array;
  v.items_ = std::move(items);
  return v;
}
Value Value::make_object(std::vector<Member> members) {
  Value v;
  v.kind_ = Kind::Object;
  v.members_ = std::move(members);
  return v;
}

const Value* Value::find(std::string_view key) const {
  if (kind_ != Kind::Object) return nullptr;
  for (const auto& [k, v] : members_) {
    if (k == key) return &v;
  }
  return nullptr;
}

std::optional<std::uint64_t> Value::as_u64() const {
  if (kind_ != Kind::Number) return std::nullopt;
  std::uint64_t out = 0;
  const char* first = text_.data();
  const char* last = first + text_.size();
  auto [ptr, ec] = std::from_chars(first, last, out);
  if (ec != std::errc() || ptr != last) return std::nullopt;  // fraction, exponent, sign or overflow
  return out;
}

std::optional<std::int64_t> Value::as_i64() const {
  if (kind_ != Kind::Number) return std::nullopt;
  std::int64_t out = 0;
  const char* first = text_.data();
  const char* last = first + text_.size();
  auto [ptr, ec] = std::from_chars(first, last, out);
  if (ec != std::errc() || ptr != last) return std::nullopt;
  return out;
}

std::optional<double> Value::as_double() const {
  if (kind_ != Kind::Number) return std::nullopt;
  // strtod is locale-dependent in theory; JSON numbers never contain a locale
  // separator, and the "C" locale is the default for a program that never
  // calls setlocale.
  char* end = nullptr;
  const double d = std::strtod(text_.c_str(), &end);
  if (end != text_.c_str() + text_.size() || !std::isfinite(d)) return std::nullopt;
  return d;
}

namespace {

class Parser {
 public:
  Parser(std::string_view in, const Limits& limits) : in_(in), limits_(limits) {}

  Result<Value> run() {
    if (!utf8::valid(in_)) return fail("json: input is not valid UTF-8");
    skip_ws();
    auto v = value(0);
    if (!v) return v;
    skip_ws();
    if (pos_ != in_.size()) return error("unexpected data after the top-level value");
    return v;
  }

 private:
  std::string_view in_;
  const Limits& limits_;
  std::size_t pos_ = 0;
  std::size_t nodes_ = 0;

  [[nodiscard]] Error error(const std::string& what) const {
    return fail("json: " + what + " at byte " + std::to_string(pos_));
  }

  void skip_ws() {
    while (pos_ < in_.size()) {
      const char c = in_[pos_];
      if (c != ' ' && c != '\t' && c != '\n' && c != '\r') break;
      ++pos_;
    }
  }

  bool consume(std::string_view lit) {
    if (in_.substr(pos_, lit.size()) != lit) return false;
    pos_ += lit.size();
    return true;
  }

  // Recursion is bounded by Limits::max_depth (checked in object() and array()).
  Result<Value> value(std::size_t depth) {  // NOLINT(misc-no-recursion)
    if (++nodes_ > limits_.max_nodes) return error("too many values");
    if (pos_ >= in_.size()) return error("unexpected end of input");
    switch (in_[pos_]) {
      case '{':
        return object(depth + 1);
      case '[':
        return array(depth + 1);
      case '"': {
        auto s = string();
        if (!s) return Error{s.error()};
        return Value::make_string(std::move(s.value()));
      }
      case 't':
        if (consume("true")) return Value::make_bool(true);
        return error("invalid literal");
      case 'f':
        if (consume("false")) return Value::make_bool(false);
        return error("invalid literal");
      case 'n':
        if (consume("null")) return Value{};
        return error("invalid literal");
      default:
        return number();
    }
  }

  Result<Value> object(std::size_t depth) {  // NOLINT(misc-no-recursion)
    if (depth > limits_.max_depth) return error("nesting too deep");
    ++pos_;  // '{'
    std::vector<Value::Member> members;
    std::unordered_set<std::string> seen;
    skip_ws();
    if (pos_ < in_.size() && in_[pos_] == '}') {
      ++pos_;
      return Value::make_object(std::move(members));
    }
    while (true) {
      skip_ws();
      if (pos_ >= in_.size() || in_[pos_] != '"') return error("expected a string key");
      auto key = string();
      if (!key) return Error{key.error()};
      if (!seen.insert(key.value()).second) return error("duplicate key \"" + key.value() + "\"");
      skip_ws();
      if (pos_ >= in_.size() || in_[pos_] != ':') return error("expected ':'");
      ++pos_;
      skip_ws();
      auto v = value(depth);
      if (!v) return v;
      members.emplace_back(std::move(key.value()), std::move(v.value()));
      skip_ws();
      if (pos_ >= in_.size()) return error("unterminated object");
      if (in_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (in_[pos_] == '}') {
        ++pos_;
        return Value::make_object(std::move(members));
      }
      return error("expected ',' or '}'");
    }
  }

  Result<Value> array(std::size_t depth) {  // NOLINT(misc-no-recursion)
    if (depth > limits_.max_depth) return error("nesting too deep");
    ++pos_;  // '['
    std::vector<Value> items;
    skip_ws();
    if (pos_ < in_.size() && in_[pos_] == ']') {
      ++pos_;
      return Value::make_array(std::move(items));
    }
    while (true) {
      skip_ws();
      auto v = value(depth);
      if (!v) return v;
      items.push_back(std::move(v.value()));
      skip_ws();
      if (pos_ >= in_.size()) return error("unterminated array");
      if (in_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (in_[pos_] == ']') {
        ++pos_;
        return Value::make_array(std::move(items));
      }
      return error("expected ',' or ']'");
    }
  }

  std::optional<unsigned> hex4() {
    if (in_.size() - pos_ < 4) return std::nullopt;
    unsigned v = 0;
    for (int k = 0; k < 4; ++k) {
      const char c = in_[pos_++];
      v <<= 4U;
      if (c >= '0' && c <= '9') {
        v |= static_cast<unsigned>(c - '0');
      } else if (c >= 'a' && c <= 'f') {
        v |= static_cast<unsigned>(c - 'a' + 10);
      } else if (c >= 'A' && c <= 'F') {
        v |= static_cast<unsigned>(c - 'A' + 10);
      } else {
        return std::nullopt;
      }
    }
    return v;
  }

  Result<std::string> string() {
    ++pos_;  // opening quote
    std::string out;
    while (true) {
      if (pos_ >= in_.size()) return error("unterminated string");
      const char c = in_[pos_];
      if (c == '"') {
        ++pos_;
        return out;
      }
      if (static_cast<unsigned char>(c) < 0x20) return error("control character in string");
      if (c != '\\') {
        out.push_back(c);  // input is already known to be valid UTF-8
        ++pos_;
        continue;
      }
      ++pos_;
      if (pos_ >= in_.size()) return error("unterminated escape");
      const char e = in_[pos_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          auto hi = hex4();
          if (!hi) return error("bad \\u escape");
          char32_t cp = *hi;
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (!consume("\\u")) return error("unpaired surrogate");
            auto lo = hex4();
            if (!lo || *lo < 0xDC00 || *lo > 0xDFFF) return error("unpaired surrogate");
            cp = 0x10000 + ((cp - 0xD800) << 10U) + (*lo - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            return error("unpaired surrogate");
          }
          utf8::append(out, cp);
          break;
        }
        default:
          return error("invalid escape");
      }
    }
  }

  Result<Value> number() {
    const std::size_t start = pos_;
    auto digit = [&] { return pos_ < in_.size() && in_[pos_] >= '0' && in_[pos_] <= '9'; };
    if (pos_ < in_.size() && in_[pos_] == '-') ++pos_;
    if (!digit()) return error("invalid value");
    if (in_[pos_] == '0') {
      ++pos_;
    } else {
      while (digit()) ++pos_;
    }
    if (pos_ < in_.size() && in_[pos_] == '.') {
      ++pos_;
      if (!digit()) return error("invalid number");
      while (digit()) ++pos_;
    }
    if (pos_ < in_.size() && (in_[pos_] == 'e' || in_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < in_.size() && (in_[pos_] == '+' || in_[pos_] == '-')) ++pos_;
      if (!digit()) return error("invalid number");
      while (digit()) ++pos_;
    }
    return Value::make_number(std::string(in_.substr(start, pos_ - start)));
  }
};

}  // namespace

Result<Value> parse(std::string_view input, const Limits& limits) {
  return Parser(input, limits).run();
}

void append_quoted(std::string& out, std::string_view s) {
  static constexpr char kHex[] = "0123456789abcdef";
  out.push_back('"');
  for (const char c : s) {
    const auto u = static_cast<unsigned char>(c);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (u < 0x20) {
          out += "\\u00";
          out.push_back(kHex[u >> 4U]);
          out.push_back(kHex[u & 0xFU]);
        } else {
          out.push_back(c);
        }
    }
  }
  out.push_back('"');
}

}  // namespace llmi::json
