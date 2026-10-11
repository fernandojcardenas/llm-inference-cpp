#include "llmi/grammar/grammar.hpp"

#include <algorithm>
#include <unordered_map>
#include <utility>

#include "llmi/util/utf8.hpp"

namespace llmi::grammar {

bool CharSet::contains(char32_t cp) const {
  const bool in = std::any_of(ranges.begin(), ranges.end(), [cp](const Range& r) { return cp >= r.lo && cp <= r.hi; });
  return in != negated;
}

bool CharSet::intersects(char32_t lo, char32_t hi) const {
  if (lo > hi) return false;
  if (!negated) {
    return std::any_of(ranges.begin(), ranges.end(), [&](const Range& r) { return r.lo <= hi && r.hi >= lo; });
  }
  // Negated: is some code point in [lo, hi] left uncovered by the ranges?
  std::vector<Range> sorted = ranges;
  std::sort(sorted.begin(), sorted.end(), [](const Range& a, const Range& b) { return a.lo < b.lo; });
  std::uint64_t cur = lo;  // 64-bit: cur may step one past 0x10FFFF
  for (const Range& r : sorted) {
    if (r.hi < cur) continue;
    if (r.lo > cur) return true;  // gap at cur, and cur <= hi here
    cur = static_cast<std::uint64_t>(r.hi) + 1;
    if (cur > hi) return false;
  }
  return cur <= hi;
}

std::size_t Grammar::symbol_count() const {
  std::size_t n = 0;
  for (const Rule& r : rules_) {
    for (const Sequence& s : r.alternatives) n += s.size();
  }
  return n;
}

namespace {

constexpr char32_t kMaxCodePoint = 0x10FFFF;

bool is_name_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
}

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

}  // namespace

// Recursive-descent parser for the GBNF subset described in grammar.hpp.
// Repetitions and groups are desugared into generated rules as they are
// parsed, so the result only ever contains plain sequences of CharSets and
// rule references -- the one shape the matcher understands.
class Parser {
 public:
  Parser(std::string_view text, const Limits& limits) : text_(text), limits_(limits) {}

  Result<Grammar> run() {
    if (text_.size() > limits_.max_grammar_bytes) {
      return fail("grammar is " + std::to_string(text_.size()) + " bytes, over the limit of " +
                  std::to_string(limits_.max_grammar_bytes));
    }
    if (!utf8::valid(text_)) return fail("grammar is not valid UTF-8");

    skip_space();
    if (pos_ >= text_.size()) return fail("grammar is empty");
    while (pos_ < text_.size()) {
      if (!parse_rule()) return fail(error_);
      skip_space();
    }

    for (std::size_t i = 0; i < rules_.size(); ++i) {
      if (!defined_[i]) return fail("rule '" + rules_[i].name + "' is used but never defined");
    }
    const auto root = names_.find("root");
    if (root == names_.end()) return fail("grammar has no 'root' rule");

    std::size_t symbols = 0;
    for (const Rule& r : rules_) {
      for (const Sequence& s : r.alternatives) symbols += s.size();
    }
    if (symbols > limits_.max_symbols) {
      return fail("grammar expands to " + std::to_string(symbols) + " symbols, over the limit of " +
                  std::to_string(limits_.max_symbols));
    }
    if (!check_left_recursion()) return fail(error_);

    Grammar g;
    g.rules_ = std::move(rules_);
    g.root_ = root->second;
    return g;
  }

 private:
  // --- lexing -------------------------------------------------------------

  void skip_space() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else if (c == '#') {
        while (pos_ < text_.size() && text_[pos_] != '\n') ++pos_;
      } else {
        break;
      }
    }
  }

  std::string read_name() {
    const std::size_t start = pos_;
    while (pos_ < text_.size() && is_name_char(text_[pos_])) ++pos_;
    return std::string(text_.substr(start, pos_ - start));
  }

  // True if the text at pos_ is "name ::=": the start of the next rule.
  bool at_rule_start() {
    const std::size_t save = pos_;
    const std::string name = read_name();
    skip_space();
    const bool yes = !name.empty() && text_.substr(pos_, 3) == "::=";
    pos_ = save;
    return yes;
  }

  bool error(std::string message) {
    error_ = std::move(message) + " (at byte " + std::to_string(pos_) + ")";
    return false;
  }

  // Reads one character of a literal or class at pos_, handling escapes.
  bool read_char(char32_t& out) {
    if (pos_ >= text_.size()) return error("unexpected end of grammar");
    if (text_[pos_] != '\\') {
      auto cp = utf8::next(text_, pos_);
      if (!cp) return error("invalid UTF-8");  // unreachable: validated up front
      out = *cp;
      return true;
    }
    ++pos_;
    if (pos_ >= text_.size()) return error("unfinished escape");
    const char e = text_[pos_++];
    std::size_t hex_digits = 0;
    switch (e) {
      case 'n': out = '\n'; return true;
      case 'r': out = '\r'; return true;
      case 't': out = '\t'; return true;
      case '\\': case '"': case '[': case ']': case '-': case '^': case '/': out = static_cast<char32_t>(e); return true;
      case 'x': hex_digits = 2; break;
      case 'u': hex_digits = 4; break;
      case 'U': hex_digits = 8; break;
      default: return error(std::string("unknown escape '\\") + e + "'");
    }
    std::uint32_t v = 0;
    for (std::size_t i = 0; i < hex_digits; ++i) {
      if (pos_ >= text_.size()) return error("unfinished hex escape");
      const int h = hex_value(text_[pos_++]);
      if (h < 0) return error("bad hex digit in escape");
      v = (v << 4U) | static_cast<std::uint32_t>(h);
    }
    if (v > kMaxCodePoint || (v >= 0xD800 && v <= 0xDFFF)) return error("escape is not a valid code point");
    out = v;
    return true;
  }

  // --- rules ---------------------------------------------------------------

  // Index of the rule called `name`, creating an undefined placeholder on
  // first reference so rules may be used before they are defined.
  bool rule_index(const std::string& name, std::uint32_t& out) {
    const auto it = names_.find(name);
    if (it != names_.end()) {
      out = it->second;
      return true;
    }
    if (!new_rule(name, out)) return false;
    names_.emplace(name, out);
    return true;
  }

  bool new_rule(std::string name, std::uint32_t& out) {
    if (rules_.size() >= limits_.max_rules) {
      return error("grammar needs more than " + std::to_string(limits_.max_rules) + " rules");
    }
    out = static_cast<std::uint32_t>(rules_.size());
    rules_.push_back(Rule{std::move(name), {}});
    defined_.push_back(false);
    return true;
  }

  bool generated_rule(std::uint32_t parent, std::vector<Sequence> alts, std::uint32_t& out) {
    std::string name = rules_[parent].name + "_" + std::to_string(++generated_);
    if (!new_rule(std::move(name), out)) return false;
    rules_[out].alternatives = std::move(alts);
    defined_[out] = true;
    return count_symbols(rules_[out].alternatives);
  }

  bool count_symbols(const std::vector<Sequence>& alts) {
    for (const Sequence& s : alts) symbols_ += s.size();
    if (symbols_ > limits_.max_symbols) {
      return error("grammar expands to more than " + std::to_string(limits_.max_symbols) + " symbols");
    }
    return true;
  }

  bool parse_rule() {
    const std::string name = read_name();
    if (name.empty()) return error("expected a rule name");
    skip_space();
    if (text_.substr(pos_, 3) != "::=") return error("expected '::=' after rule name '" + name + "'");
    pos_ += 3;
    std::uint32_t index = 0;
    if (!rule_index(name, index)) return false;
    if (defined_[index]) return error("rule '" + name + "' is defined twice");
    defined_[index] = true;  // set first: the body may refer to itself
    std::vector<Sequence> alts;
    if (!parse_alternatives(index, 0, alts)) return false;
    if (!count_symbols(alts)) return false;
    rules_[index].alternatives = std::move(alts);
    return true;
  }

  // The three functions below recurse through groups; depth is bounded by
  // Limits::max_nesting (checked in parse_primary before every '(').
  // NOLINTNEXTLINE(misc-no-recursion)
  bool parse_alternatives(std::uint32_t rule, std::size_t depth, std::vector<Sequence>& out) {
    while (true) {
      Sequence seq;
      if (!parse_sequence(rule, depth, seq)) return false;
      out.push_back(std::move(seq));
      skip_space();
      if (pos_ < text_.size() && text_[pos_] == '|') {
        ++pos_;
        continue;
      }
      return true;
    }
  }

  // NOLINTNEXTLINE(misc-no-recursion): bounded by max_nesting, see parse_alternatives
  bool parse_sequence(std::uint32_t rule, std::size_t depth, Sequence& out) {
    while (true) {
      skip_space();
      if (pos_ >= text_.size()) return true;
      const char c = text_[pos_];
      if (c == '|' || c == ')') return true;
      if (is_name_char(c) && at_rule_start()) return true;

      std::vector<Symbol> unit;
      if (!parse_primary(rule, depth, unit)) return false;
      if (!parse_postfix(rule, unit)) return false;
      out.insert(out.end(), unit.begin(), unit.end());
    }
  }

  // NOLINTNEXTLINE(misc-no-recursion): bounded by max_nesting, see parse_alternatives
  bool parse_primary(std::uint32_t rule, std::size_t depth, std::vector<Symbol>& out) {
    const char c = text_[pos_];
    if (c == '"') {
      ++pos_;
      while (true) {
        if (pos_ >= text_.size()) return error("unterminated string literal");
        if (text_[pos_] == '"') {
          ++pos_;
          break;
        }
        char32_t cp = 0;
        if (!read_char(cp)) return false;
        Symbol s;
        s.chars.ranges.push_back({cp, cp});
        out.push_back(std::move(s));
      }
      return true;  // "" is a valid empty literal
    }
    if (c == '[') {
      ++pos_;
      Symbol s;
      if (pos_ < text_.size() && text_[pos_] == '^') {
        s.chars.negated = true;
        ++pos_;
      }
      while (true) {
        if (pos_ >= text_.size()) return error("unterminated character class");
        if (text_[pos_] == ']') {
          ++pos_;
          break;
        }
        char32_t lo = 0;
        if (!read_char(lo)) return false;
        char32_t hi = lo;
        if (pos_ + 1 < text_.size() && text_[pos_] == '-' && text_[pos_ + 1] != ']') {
          ++pos_;
          if (!read_char(hi)) return false;
          if (hi < lo) return error("character range is backwards");
        }
        s.chars.ranges.push_back({lo, hi});
      }
      if (s.chars.ranges.empty() && !s.chars.negated) return error("empty character class matches nothing");
      out.push_back(std::move(s));
      return true;
    }
    if (c == '.') {
      ++pos_;
      Symbol s;
      s.chars.negated = true;  // the complement of nothing: any code point
      out.push_back(std::move(s));
      return true;
    }
    if (c == '(') {
      if (depth + 1 > limits_.max_nesting) return error("groups nested too deeply");
      ++pos_;
      std::vector<Sequence> alts;
      if (!parse_alternatives(rule, depth + 1, alts)) return false;
      skip_space();
      if (pos_ >= text_.size() || text_[pos_] != ')') return error("expected ')'");
      ++pos_;
      std::uint32_t g = 0;
      if (!generated_rule(rule, std::move(alts), g)) return false;
      out.push_back(rule_ref(g));
      return true;
    }
    if (is_name_char(c)) {
      const std::string name = read_name();
      std::uint32_t index = 0;
      if (!rule_index(name, index)) return false;
      out.push_back(rule_ref(index));
      return true;
    }
    return error(std::string("unexpected character '") + c + "'");
  }

  static Symbol rule_ref(std::uint32_t index) {
    Symbol s;
    s.kind = Symbol::Kind::Rule;
    s.rule = index;
    return s;
  }

  bool read_count(std::size_t& out) {
    const std::size_t start = pos_;
    std::size_t v = 0;
    while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
      v = v * 10 + static_cast<std::size_t>(text_[pos_] - '0');
      if (v > limits_.max_repetition) return error("repetition count over " + std::to_string(limits_.max_repetition));
      ++pos_;
    }
    if (pos_ == start) return error("expected a number in {m,n}");
    out = v;
    return true;
  }

  // Applies a postfix *, +, ?, {m}, {m,} or {m,n} to `unit`, replacing it
  // with its desugared form.
  bool parse_postfix(std::uint32_t rule, std::vector<Symbol>& unit) {
    if (pos_ >= text_.size()) return true;
    std::size_t min = 0;
    std::size_t max = 0;
    bool unbounded = false;
    const char c = text_[pos_];
    if (c == '*') {
      unbounded = true;
    } else if (c == '+') {
      min = 1;
      unbounded = true;
    } else if (c == '?') {
      max = 1;
    } else if (c == '{') {
      ++pos_;
      skip_space();
      if (!read_count(min)) return false;
      skip_space();
      if (pos_ < text_.size() && text_[pos_] == ',') {
        ++pos_;
        skip_space();
        if (pos_ < text_.size() && text_[pos_] == '}') {
          unbounded = true;
        } else {
          if (!read_count(max)) return false;
          if (max < min) return error("{m,n} with n < m");
        }
      } else {
        max = min;
      }
      skip_space();
      if (pos_ >= text_.size() || text_[pos_] != '}') return error("expected '}'");
    } else {
      return true;
    }
    ++pos_;

    // Repeat a single symbol; a multi-symbol unit (a literal) is wrapped in
    // its own rule first.
    Symbol item;
    if (unit.size() == 1) {
      item = unit.front();
    } else {
      std::uint32_t g = 0;
      if (!generated_rule(rule, {unit}, g)) return false;
      item = rule_ref(g);
    }

    std::vector<Symbol> out(min, item);
    if (unbounded) {
      // star ::= item star | (empty)
      std::uint32_t star = 0;
      if (!generated_rule(rule, {}, star)) return false;
      rules_[star].alternatives = {Sequence{item, rule_ref(star)}, Sequence{}};
      if (!count_symbols({Sequence{item, rule_ref(star)}})) return false;
      out.push_back(rule_ref(star));
    } else if (max > min) {
      // opt_1 ::= item | (empty);  opt_k ::= item opt_{k-1} | (empty)
      std::uint32_t prev = 0;
      if (!generated_rule(rule, {Sequence{item}, Sequence{}}, prev)) return false;
      for (std::size_t k = 2; k <= max - min; ++k) {
        std::uint32_t next = 0;
        if (!generated_rule(rule, {Sequence{item, rule_ref(prev)}, Sequence{}}, next)) return false;
        prev = next;
      }
      out.push_back(rule_ref(prev));
    }
    if (out.size() > limits_.max_symbols) return error("repetition expands past the symbol limit");
    unit = std::move(out);
    return true;
  }

  // --- checks --------------------------------------------------------------

  // Left recursion (a rule that can reach itself again without consuming a
  // character) would make the matcher expand forever, so it is rejected.
  bool check_left_recursion() {
    const std::size_t n = rules_.size();
    std::vector<bool> nullable(n, false);
    for (bool changed = true; changed;) {
      changed = false;
      for (std::size_t r = 0; r < n; ++r) {
        if (nullable[r]) continue;
        for (const Sequence& s : rules_[r].alternatives) {
          const bool all = std::all_of(s.begin(), s.end(), [&](const Symbol& sym) {
            return sym.kind == Symbol::Kind::Rule && nullable[sym.rule];
          });
          if (all) {
            nullable[r] = true;
            changed = true;
            break;
          }
        }
      }
    }
    // Edge r -> q: q can be the first thing r expands to.
    std::vector<std::vector<std::uint32_t>> first(n);
    for (std::size_t r = 0; r < n; ++r) {
      for (const Sequence& s : rules_[r].alternatives) {
        for (const Symbol& sym : s) {
          if (sym.kind != Symbol::Kind::Rule) break;
          first[r].push_back(sym.rule);
          if (!nullable[sym.rule]) break;
        }
      }
    }
    // Iterative DFS cycle check: 0 = unvisited, 1 = on the stack, 2 = done.
    std::vector<std::uint8_t> state(n, 0);
    for (std::size_t start = 0; start < n; ++start) {
      if (state[start] != 0) continue;
      std::vector<std::pair<std::size_t, std::size_t>> stack{{start, 0}};
      state[start] = 1;
      while (!stack.empty()) {
        auto& [r, i] = stack.back();
        if (i < first[r].size()) {
          const std::uint32_t q = first[r][i++];
          if (state[q] == 1) {
            return error("rule '" + rules_[q].name + "' is left-recursive (it can reach itself without consuming input)");
          }
          if (state[q] == 0) {
            state[q] = 1;
            stack.emplace_back(q, 0);
          }
        } else {
          state[r] = 2;
          stack.pop_back();
        }
      }
    }
    return true;
  }

  std::string_view text_;
  const Limits& limits_;
  std::size_t pos_ = 0;
  std::string error_;
  std::vector<Rule> rules_;
  std::vector<bool> defined_;
  std::unordered_map<std::string, std::uint32_t> names_;
  std::size_t generated_ = 0;
  std::size_t symbols_ = 0;
};

Result<Grammar> Grammar::parse(std::string_view text, const Limits& limits) { return Parser(text, limits).run(); }

}  // namespace llmi::grammar
