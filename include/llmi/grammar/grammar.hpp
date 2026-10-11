#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "llmi/util/result.hpp"

namespace llmi::grammar {

// Constrained decoding, part 1: the grammar itself (M8, ADR 0009).
//
// The input format is a subset of llama.cpp's GBNF, so a grammar written for
// llama.cpp's `--grammar` works here unchanged as long as it stays inside the
// subset:
//
//   root   ::= "{" ws pair (ws "," ws pair)* ws "}"   # a comment
//   pair   ::= key ws ":" ws [0-9]+
//   key    ::= "\"" [a-z_]{1,16} "\""
//   ws     ::= [ \t\n]{0,20}
//
//   - rules:      name ::= alternatives      (names: letters, digits, '-', '_')
//   - literals:   "text" with escapes \n \r \t \\ \" \xHH \uHHHH \UHHHHHHHH
//   - classes:    [a-z0-9_], negated [^"\\], escapes as in literals; '.' = any
//                 code point
//   - grouping:   ( ... ), alternatives with |
//   - repetition: * + ? {m} {m,} {m,n}
//
// A rule ends where the next "name ::=" begins, so alternatives may span
// lines. The start rule is always "root".
//
// Grammars reach this parser from HTTP requests (as JSON schemas converted to
// grammar text), so it is written for untrusted input, the same way M1's
// loaders were: every size it builds is bounded by Limits, and anything the
// matcher could not run safely is rejected here instead -- left recursion
// (which would expand forever) and references to undefined rules.

struct Limits {
  std::size_t max_grammar_bytes = std::size_t{64} * 1024;
  std::size_t max_rules = 4096;        // after repetitions and groups are desugared
  std::size_t max_symbols = 200'000;   // across all rules, after desugaring
  std::size_t max_repetition = 1000;   // the largest m or n in {m,n}
  std::size_t max_nesting = 64;        // nested groups
};

// A set of code points: the union of `ranges` (inclusive), or its complement
// when `negated` is set.
struct CharSet {
  struct Range {
    char32_t lo = 0;
    char32_t hi = 0;
  };
  std::vector<Range> ranges;
  bool negated = false;

  [[nodiscard]] bool contains(char32_t cp) const;
  // True if some code point in [lo, hi] is in this set.
  [[nodiscard]] bool intersects(char32_t lo, char32_t hi) const;
};

// One element of a sequence: a set of code points to match, or a reference
// to another rule.
struct Symbol {
  enum class Kind : std::uint8_t { Chars, Rule };
  Kind kind = Kind::Chars;
  CharSet chars;            // Kind::Chars
  std::uint32_t rule = 0;   // Kind::Rule
};

using Sequence = std::vector<Symbol>;

struct Rule {
  std::string name;             // generated rules are named "<parent>_<n>"
  std::vector<Sequence> alternatives;  // an empty Sequence matches the empty string
};

class Grammar {
 public:
  static Result<Grammar> parse(std::string_view text, const Limits& limits = {});

  [[nodiscard]] const std::vector<Rule>& rules() const { return rules_; }
  [[nodiscard]] std::uint32_t root() const { return root_; }
  [[nodiscard]] std::size_t symbol_count() const;

 private:
  std::vector<Rule> rules_;
  std::uint32_t root_ = 0;
  friend class Parser;
};

}  // namespace llmi::grammar
