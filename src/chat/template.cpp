#include "llmi/chat/template.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "llmi/util/json.hpp"

namespace llmi::chat {

namespace {

// ------------------------------------------------------------- values

enum class Kind : std::uint8_t { Undefined, Bool, Number, String, Message, MessageList, Loop };

struct LoopInfo {
  std::size_t index0;
  bool first;
  bool last;
};

struct Value {
  Kind kind = Kind::Undefined;
  bool b = false;
  double n = 0;
  std::string s;
  const Message* msg = nullptr;
  const std::vector<Message>* msgs = nullptr;
  const LoopInfo* loop = nullptr;

  static Value undefined() { return {}; }
  static Value boolean(bool v) {
    Value x;
    x.kind = Kind::Bool;
    x.b = v;
    return x;
  }
  static Value number(double v) {
    Value x;
    x.kind = Kind::Number;
    x.n = v;
    return x;
  }
  static Value str(std::string v) {
    Value x;
    x.kind = Kind::String;
    x.s = std::move(v);
    return x;
  }
  static Value message(const Message* m) {
    Value x;
    x.kind = Kind::Message;
    x.msg = m;
    return x;
  }
  static Value message_list(const std::vector<Message>* m) {
    Value x;
    x.kind = Kind::MessageList;
    x.msgs = m;
    return x;
  }
  static Value loop_info(const LoopInfo* l) {
    Value x;
    x.kind = Kind::Loop;
    x.loop = l;
    return x;
  }
};

bool is_truthy(const Value& v) {
  switch (v.kind) {
    case Kind::Undefined:
      return false;
    case Kind::Bool:
      return v.b;
    case Kind::Number:
      return v.n != 0;
    case Kind::String:
      return !v.s.empty();
    case Kind::Message:
    case Kind::MessageList:
    case Kind::Loop:
      return true;
  }
  return false;
}

bool values_equal(const Value& a, const Value& b) {
  if (a.kind != b.kind) return false;
  switch (a.kind) {
    case Kind::Undefined:
      return true;
    case Kind::Bool:
      return a.b == b.b;
    case Kind::Number:
      return a.n == b.n;
    case Kind::String:
      return a.s == b.s;
    default:
      return false;  // objects are never compared by this template set
  }
}

Value attr_of(const Value& base, const std::string& name) {
  if (base.kind == Kind::Message) {
    if (name == "role") return Value::str(base.msg->role);
    if (name == "content") return Value::str(base.msg->content);
    if (name == "tool_calls" || name == "function") return Value::undefined();  // tool calls are never modeled
    return Value::undefined();
  }
  if (base.kind == Kind::Loop) {
    if (name == "first") return Value::boolean(base.loop->first);
    if (name == "last") return Value::boolean(base.loop->last);
    if (name == "index0") return Value::number(static_cast<double>(base.loop->index0));
    if (name == "index") return Value::number(static_cast<double>(base.loop->index0 + 1));
    return Value::undefined();
  }
  return Value::undefined();
}

Value index_of(const Value& base, const Value& idx) {
  if (base.kind == Kind::MessageList && idx.kind == Kind::Number) {
    const auto i = static_cast<std::size_t>(idx.n);
    if (i < base.msgs->size()) return Value::message(&(*base.msgs)[i]);
    return Value::undefined();
  }
  if (base.kind == Kind::Message && idx.kind == Kind::String) return attr_of(base, idx.s);
  return Value::undefined();
}

std::string to_output_string(const Value& v) {
  switch (v.kind) {
    case Kind::String:
      return v.s;
    case Kind::Bool:
      return v.b ? "True" : "False";
    case Kind::Number: {
      // Our templates only ever print strings; this is a defensive fallback.
      std::string s = std::to_string(v.n);
      while (!s.empty() && s.back() == '0') s.pop_back();
      if (!s.empty() && s.back() == '.') s.pop_back();
      return s;
    }
    default:
      return "";  // Undefined / objects: printing them never happens on the paths this engine takes
  }
}

// ------------------------------------------------------------- expr tokens

struct Token {
  enum class Kind : std::uint8_t { Ident, String, Number, Punct, End } kind = Kind::End;
  std::string s;
  double num = 0;
};

[[noreturn]] void fail_parse(const std::string& msg) { throw std::runtime_error("chat template: " + msg); }

std::vector<Token> tokenize_expr(const std::string& src) {
  std::vector<Token> out;
  std::size_t i = 0;
  while (i < src.size()) {
    const char c = src[i];
    if (std::isspace(static_cast<unsigned char>(c)) != 0) {
      ++i;
      continue;
    }
    if (c == '\'' || c == '"') {
      const char q = c;
      std::string s;
      ++i;
      while (i < src.size() && src[i] != q) {
        if (src[i] == '\\' && i + 1 < src.size()) {
          const char e = src[i + 1];
          if (e == 'n') s += '\n';
          else if (e == 't') s += '\t';
          else s += e;  // \\, \', \" and anything else: keep the escaped character literally
          i += 2;
        } else {
          s += src[i++];
        }
      }
      if (i >= src.size()) fail_parse("unterminated string literal");
      ++i;  // closing quote
      out.push_back({Token::Kind::String, s, 0});
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) != 0) {
      std::size_t j = i;
      while (j < src.size() && (std::isdigit(static_cast<unsigned char>(src[j])) != 0 || src[j] == '.')) ++j;
      out.push_back({Token::Kind::Number, "", std::stod(src.substr(i, j - i))});
      i = j;
      continue;
    }
    if (std::isalpha(static_cast<unsigned char>(c)) != 0 || c == '_') {
      std::size_t j = i;
      while (j < src.size() && (std::isalnum(static_cast<unsigned char>(src[j])) != 0 || src[j] == '_')) ++j;
      out.push_back({Token::Kind::Ident, src.substr(i, j - i), 0});
      i = j;
      continue;
    }
    if (c == '=' && i + 1 < src.size() && src[i + 1] == '=') {
      out.push_back({Token::Kind::Punct, "==", 0});
      i += 2;
      continue;
    }
    if (c == '!' && i + 1 < src.size() && src[i + 1] == '=') {
      out.push_back({Token::Kind::Punct, "!=", 0});
      i += 2;
      continue;
    }
    if (std::string("()[].,|+=-").find(c) != std::string::npos) {
      out.push_back({Token::Kind::Punct, std::string(1, c), 0});
      ++i;
      continue;
    }
    fail_parse(std::string("unexpected character '") + c + "' in expression");
  }
  out.push_back({Token::Kind::End, "", 0});
  return out;
}

// ------------------------------------------------------------- expr AST

struct Expr {
  enum class Kind : std::uint8_t { Lit, Ident, Attr, Index, BinOp, Not, IsDefined, Filter } kind;
  Value lit;
  std::string name;  // Ident name / Attr name / BinOp op / Filter name
  std::unique_ptr<Expr> a, b;
  bool negate = false;  // IsDefined
};
using ExprPtr = std::unique_ptr<Expr>;

ExprPtr make_lit(Value v) {
  auto e = std::make_unique<Expr>();
  e->kind = Expr::Kind::Lit;
  e->lit = std::move(v);
  return e;
}

class ExprParser {
 public:
  explicit ExprParser(std::vector<Token> toks) : t_(std::move(toks)) {}

  ExprPtr parse_full() {
    auto e = parse_or();
    expect_end();
    return e;
  }

  // Parses a leading "NAME" and consumes it (for `for`/`set` statements).
  std::string parse_ident() {
    if (peek().kind != Token::Kind::Ident) fail_parse("expected a name");
    return t_[i_++].s;
  }
  bool consume_ident(const std::string& word) {
    if (peek().kind == Token::Kind::Ident && peek().s == word) {
      ++i_;
      return true;
    }
    return false;
  }
  void expect_ident(const std::string& word) {
    if (!consume_ident(word)) fail_parse("expected '" + word + "'");
  }
  ExprPtr parse_or_full() {
    auto e = parse_or();
    expect_end();
    return e;
  }
  bool consume_punct_public(const std::string& p) { return consume_punct(p); }

 private:
  std::vector<Token> t_;
  std::size_t i_ = 0;

  const Token& peek() { return t_[i_]; }
  void expect_end() {
    if (peek().kind != Token::Kind::End) fail_parse("unexpected trailing tokens");
  }
  bool consume_punct(const std::string& p) {
    if (peek().kind == Token::Kind::Punct && peek().s == p) {
      ++i_;
      return true;
    }
    return false;
  }

  ExprPtr parse_or() {  // NOLINT(misc-no-recursion)
    auto lhs = parse_and();
    while (consume_ident("or")) {
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::BinOp;
      e->name = "or";
      e->a = std::move(lhs);
      e->b = parse_and();
      lhs = std::move(e);
    }
    return lhs;
  }
  ExprPtr parse_and() {  // NOLINT(misc-no-recursion)
    auto lhs = parse_not();
    while (consume_ident("and")) {
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::BinOp;
      e->name = "and";
      e->a = std::move(lhs);
      e->b = parse_not();
      lhs = std::move(e);
    }
    return lhs;
  }
  ExprPtr parse_not() {  // NOLINT(misc-no-recursion)
    if (consume_ident("not")) {
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::Not;
      e->a = parse_not();
      return e;
    }
    return parse_compare();
  }
  ExprPtr parse_compare() { return parse_compare_impl(parse_is()); }  // NOLINT(misc-no-recursion)
  ExprPtr parse_compare_impl(ExprPtr lhs) {  // NOLINT(misc-no-recursion)
    if (peek().kind == Token::Kind::Punct && peek().s == "==") {
      ++i_;
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::BinOp;
      e->name = "==";
      e->a = std::move(lhs);
      e->b = parse_is();
      return e;
    }
    if (peek().kind == Token::Kind::Punct && peek().s == "!=") {
      ++i_;
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::BinOp;
      e->name = "!=";
      e->a = std::move(lhs);
      e->b = parse_is();
      return e;
    }
    return lhs;
  }
  ExprPtr parse_is() {  // NOLINT(misc-no-recursion)
    auto lhs = parse_concat();
    if (consume_ident("is")) {
      const bool neg = consume_ident("not");
      expect_ident("defined");
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::IsDefined;
      e->a = std::move(lhs);
      e->negate = neg;
      return e;
    }
    return lhs;
  }
  ExprPtr parse_concat() {  // NOLINT(misc-no-recursion)
    auto lhs = parse_filter();
    for (;;) {
      std::string op;
      if (consume_punct("+")) op = "+";
      else if (consume_punct("-")) op = "-";
      else break;
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::BinOp;
      e->name = op;
      e->a = std::move(lhs);
      e->b = parse_filter();
      lhs = std::move(e);
    }
    return lhs;
  }
  ExprPtr parse_filter() {  // NOLINT(misc-no-recursion)
    auto lhs = parse_postfix();
    while (consume_punct("|")) {
      const std::string name = parse_ident();
      if (consume_punct("(")) {
        // Skip arguments: not needed by any filter this engine evaluates.
        int depth = 1;
        while (depth > 0) {
          if (peek().kind == Token::Kind::End) fail_parse("unterminated filter arguments");
          if (peek().kind == Token::Kind::Punct && peek().s == "(") ++depth;
          if (peek().kind == Token::Kind::Punct && peek().s == ")") --depth;
          ++i_;
        }
      }
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::Filter;
      e->name = name;
      e->a = std::move(lhs);
      lhs = std::move(e);
    }
    return lhs;
  }
  ExprPtr parse_postfix() {  // NOLINT(misc-no-recursion)
    auto lhs = parse_primary();
    for (;;) {
      if (consume_punct(".")) {
        const std::string name = parse_ident();
        auto e = std::make_unique<Expr>();
        e->kind = Expr::Kind::Attr;
        e->name = name;
        e->a = std::move(lhs);
        lhs = std::move(e);
      } else if (consume_punct("[")) {
        auto idx = parse_or();
        if (!consume_punct("]")) fail_parse("expected ']'");
        auto e = std::make_unique<Expr>();
        e->kind = Expr::Kind::Index;
        e->a = std::move(lhs);
        e->b = std::move(idx);
        lhs = std::move(e);
      } else {
        break;
      }
    }
    return lhs;
  }
  ExprPtr parse_primary() {  // NOLINT(misc-no-recursion)
    const Token tok = peek();
    if (tok.kind == Token::Kind::String) {
      ++i_;
      return make_lit(Value::str(tok.s));
    }
    if (tok.kind == Token::Kind::Number) {
      ++i_;
      return make_lit(Value::number(tok.num));
    }
    if (consume_punct("(")) {
      auto e = parse_or();
      if (!consume_punct(")")) fail_parse("expected ')'");
      return e;
    }
    if (tok.kind == Token::Kind::Ident) {
      if (tok.s == "true" || tok.s == "True") {
        ++i_;
        return make_lit(Value::boolean(true));
      }
      if (tok.s == "false" || tok.s == "False") {
        ++i_;
        return make_lit(Value::boolean(false));
      }
      if (tok.s == "none" || tok.s == "None") {
        ++i_;
        return make_lit(Value::undefined());
      }
      ++i_;
      auto e = std::make_unique<Expr>();
      e->kind = Expr::Kind::Ident;
      e->name = tok.s;
      return e;
    }
    fail_parse("unexpected token in expression");
  }
};

struct Env {
  const std::vector<Message>* messages = nullptr;
  bool add_generation_prompt = false;
  std::map<std::string, Value> vars;
  std::vector<LoopInfo> loop_stack;
};

Value eval(const Expr& e, Env& env) {  // NOLINT(misc-no-recursion)
  switch (e.kind) {
    case Expr::Kind::Lit:
      return e.lit;
    case Expr::Kind::Ident: {
      if (e.name == "messages") return Value::message_list(env.messages);
      if (e.name == "add_generation_prompt") return Value::boolean(env.add_generation_prompt);
      if (e.name == "loop") return env.loop_stack.empty() ? Value::undefined() : Value::loop_info(&env.loop_stack.back());
      if (e.name == "tools") return Value::undefined();  // this engine never supplies tools
      const auto it = env.vars.find(e.name);
      return it != env.vars.end() ? it->second : Value::undefined();
    }
    case Expr::Kind::Attr:
      return attr_of(eval(*e.a, env), e.name);
    case Expr::Kind::Index:
      return index_of(eval(*e.a, env), eval(*e.b, env));
    case Expr::Kind::Not:
      return Value::boolean(!is_truthy(eval(*e.a, env)));
    case Expr::Kind::IsDefined: {
      const bool defined = eval(*e.a, env).kind != Kind::Undefined;
      return Value::boolean(e.negate ? !defined : defined);
    }
    case Expr::Kind::Filter:
      // "tojson" and friends are only reached on tool-calling branches this
      // engine never takes; evaluate the operand for side-effect-free safety
      // and fall back to its string form.
      return Value::str(to_output_string(eval(*e.a, env)));
    case Expr::Kind::BinOp: {
      if (e.name == "and") return Value::boolean(is_truthy(eval(*e.a, env)) && is_truthy(eval(*e.b, env)));
      if (e.name == "or") return Value::boolean(is_truthy(eval(*e.a, env)) || is_truthy(eval(*e.b, env)));
      if (e.name == "==") return Value::boolean(values_equal(eval(*e.a, env), eval(*e.b, env)));
      if (e.name == "!=") return Value::boolean(!values_equal(eval(*e.a, env), eval(*e.b, env)));
      if (e.name == "+") {
        const Value a = eval(*e.a, env);
        const Value b = eval(*e.b, env);
        if (a.kind == Kind::Number && b.kind == Kind::Number) return Value::number(a.n + b.n);
        return Value::str(to_output_string(a) + to_output_string(b));
      }
      if (e.name == "-") {
        const Value a = eval(*e.a, env);
        const Value b = eval(*e.b, env);
        return Value::number(a.n - b.n);  // only reached on index arithmetic (loop.index0 - 1); both are numbers
      }
      fail_parse("unknown operator " + e.name);
    }
  }
  fail_parse("unreachable expression kind");
}

}  // namespace

// ------------------------------------------------------------- template AST

struct ChatTemplate::Node {
  enum class Kind : std::uint8_t { Text, Output, For, If, Set } kind;
  std::string text;                   // Text
  ExprPtr expr;                       // Output / Set (rhs)
  std::string var_name;                // For / Set (lhs)
  ExprPtr iterable;                    // For
  std::vector<Node> body;              // For body
  std::vector<std::pair<ExprPtr, std::vector<Node>>> branches;  // If: cond==nullptr means else
};

namespace {

using Node = ChatTemplate::Node;

struct Piece {
  enum class Kind : std::uint8_t { Text, Output, Stmt } kind;
  std::string content;
  bool trim_left = false;
  bool trim_right = false;
};

void rstrip(std::string& s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) s.pop_back();
}
void lstrip(std::string& s) {
  std::size_t i = 0;
  while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])) != 0) ++i;
  s.erase(0, i);
}
std::string trim_copy(const std::string& s) {
  std::size_t a = 0;
  std::size_t b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a])) != 0) ++a;
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])) != 0) --b;
  return s.substr(a, b - a);
}

std::vector<Piece> lex(const std::string& src) {
  std::vector<Piece> out;
  std::size_t i = 0;
  while (i < src.size()) {
    const std::size_t open_out = src.find("{{", i);
    const std::size_t open_stmt = src.find("{%", i);
    const std::size_t open = std::min(open_out, open_stmt);
    if (open == std::string::npos) {
      out.push_back({Piece::Kind::Text, src.substr(i), false, false});
      break;
    }
    if (open > i) out.push_back({Piece::Kind::Text, src.substr(i, open - i), false, false});
    const bool is_output = open == open_out;
    std::size_t start = open + 2;
    const bool trim_left = start < src.size() && src[start] == '-';
    if (trim_left) ++start;
    const std::string close_delim = is_output ? "}}" : "%}";
    const std::size_t close = src.find(close_delim, start);
    if (close == std::string::npos) fail_parse("unterminated tag");
    const bool trim_right = close > start && src[close - 1] == '-';
    const std::size_t content_end = trim_right ? close - 1 : close;
    out.push_back({is_output ? Piece::Kind::Output : Piece::Kind::Stmt, trim_copy(src.substr(start, content_end - start)),
                   trim_left, trim_right});
    i = close + close_delim.size();
  }
  for (std::size_t k = 0; k < out.size(); ++k) {
    if (out[k].kind == Piece::Kind::Text) continue;
    if (out[k].trim_left && k > 0 && out[k - 1].kind == Piece::Kind::Text) rstrip(out[k - 1].content);
    if (out[k].trim_right && k + 1 < out.size() && out[k + 1].kind == Piece::Kind::Text) lstrip(out[k + 1].content);
  }
  return out;
}

class TemplateParser {
 public:
  explicit TemplateParser(std::vector<Piece> pieces) : p_(std::move(pieces)) {}

  std::vector<Node> parse_top() {
    std::string stopped;
    auto nodes = parse_until(&stopped);
    if (!stopped.empty()) fail_parse("unexpected '" + stopped + "'");
    return nodes;
  }

 private:
  std::vector<Piece> p_;
  std::size_t i_ = 0;

  static std::string first_word(const std::string& content) {
    std::size_t j = 0;
    while (j < content.size() && (std::isalnum(static_cast<unsigned char>(content[j])) != 0 || content[j] == '_')) ++j;
    return content.substr(0, j);
  }
  static std::string rest_after(const std::string& content, const std::string& word) {
    return trim_copy(content.substr(word.size()));
  }

  // Parses nodes until a stop keyword (endfor/endif/elif/else) or end of
  // input; *stopped is set to that keyword ("" at end of input), and the
  // stopping tag is left unconsumed so the caller can inspect/consume it.
  std::vector<Node> parse_until(std::string* stopped) {  // NOLINT(misc-no-recursion)
    std::vector<Node> nodes;
    while (i_ < p_.size()) {
      const Piece& piece = p_[i_];
      if (piece.kind == Piece::Kind::Text) {
        Node n;
        n.kind = Node::Kind::Text;
        n.text = piece.content;
        nodes.push_back(std::move(n));
        ++i_;
        continue;
      }
      if (piece.kind == Piece::Kind::Output) {
        Node n;
        n.kind = Node::Kind::Output;
        n.expr = ExprParser(tokenize_expr(piece.content)).parse_full();
        nodes.push_back(std::move(n));
        ++i_;
        continue;
      }
      const std::string word = first_word(piece.content);
      if (word == "endfor" || word == "endif" || word == "elif" || word == "else") {
        *stopped = word;
        return nodes;
      }
      if (word == "for") {
        ++i_;
        ExprParser ep(tokenize_expr(rest_after(piece.content, word)));
        const std::string var = ep.parse_ident();
        ep.expect_ident("in");
        auto iterable = ep.parse_or_full();
        std::string s;
        auto body = parse_until(&s);
        if (s != "endfor") fail_parse("expected 'endfor'");
        ++i_;  // consume endfor
        Node n;
        n.kind = Node::Kind::For;
        n.var_name = var;
        n.iterable = std::move(iterable);
        n.body = std::move(body);
        nodes.push_back(std::move(n));
        continue;
      }
      if (word == "if") {
        ++i_;
        Node n;
        n.kind = Node::Kind::If;
        auto cond = ExprParser(tokenize_expr(rest_after(piece.content, word))).parse_or_full();
        std::string s;
        auto body = parse_until(&s);
        n.branches.emplace_back(std::move(cond), std::move(body));
        while (s == "elif") {
          const Piece& ep_piece = p_[i_];
          auto econd = ExprParser(tokenize_expr(rest_after(ep_piece.content, "elif"))).parse_or_full();
          ++i_;
          auto ebody = parse_until(&s);
          n.branches.emplace_back(std::move(econd), std::move(ebody));
        }
        if (s == "else") {
          ++i_;
          auto ebody = parse_until(&s);
          n.branches.emplace_back(nullptr, std::move(ebody));
        }
        if (s != "endif") fail_parse("expected 'endif'");
        ++i_;  // consume endif
        nodes.push_back(std::move(n));
        continue;
      }
      if (word == "set") {
        ++i_;
        ExprParser ep(tokenize_expr(rest_after(piece.content, word)));
        const std::string var = ep.parse_ident();
        if (!ep.consume_punct_public("=")) fail_parse("expected '=' in 'set'");
        auto rhs = ep.parse_or_full();
        Node n;
        n.kind = Node::Kind::Set;
        n.var_name = var;
        n.expr = std::move(rhs);
        nodes.push_back(std::move(n));
        continue;
      }
      fail_parse("unsupported tag '" + word + "'");
    }
    *stopped = "";
    return nodes;
  }
};

std::string render_nodes(const std::vector<Node>& nodes, Env& env) {  // NOLINT(misc-no-recursion)
  std::string out;
  for (const Node& n : nodes) {
    switch (n.kind) {
      case Node::Kind::Text:
        out += n.text;
        break;
      case Node::Kind::Output:
        out += to_output_string(eval(*n.expr, env));
        break;
      case Node::Kind::Set:
        env.vars[n.var_name] = eval(*n.expr, env);
        break;
      case Node::Kind::For: {
        const Value iterable = eval(*n.iterable, env);
        if (iterable.kind == Kind::MessageList) {
          const std::size_t count = iterable.msgs->size();
          for (std::size_t idx = 0; idx < count; ++idx) {
            const LoopInfo li{idx, idx == 0, idx + 1 == count};
            env.loop_stack.push_back(li);
            env.vars[n.var_name] = Value::message(&(*iterable.msgs)[idx]);
            out += render_nodes(n.body, env);
            env.loop_stack.pop_back();
          }
        }
        // Any other iterable (this engine's always-undefined `tools`, or a
        // message's always-undefined `tool_calls`) yields zero iterations,
        // matching a reference call made without tools.
        env.vars.erase(n.var_name);
        break;
      }
      case Node::Kind::If: {
        for (const auto& [cond, body] : n.branches) {
          if (cond == nullptr || is_truthy(eval(*cond, env))) {
            out += render_nodes(body, env);
            break;
          }
        }
        break;
      }
    }
  }
  return out;
}

}  // namespace

Result<ChatTemplate> ChatTemplate::parse(const std::string& jinja_source) {
  try {
    // Jinja2's default environment (keep_trailing_newline=False, what Hugging
    // Face's apply_chat_template uses) drops exactly one trailing newline
    // from the template *source* before parsing, not from its output.
    std::string src = jinja_source;
    if (!src.empty() && src.back() == '\n') {
      src.pop_back();
      if (!src.empty() && src.back() == '\r') src.pop_back();
    }
    auto nodes = TemplateParser(lex(src)).parse_top();
    ChatTemplate t;
    t.root_ = std::make_shared<const std::vector<Node>>(std::move(nodes));
    return t;
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

Result<ChatTemplate> ChatTemplate::load(const std::string& tokenizer_config_path) {
  const std::ifstream f(tokenizer_config_path, std::ios::binary);
  if (!f) return fail("cannot open " + tokenizer_config_path);
  std::ostringstream ss;
  ss << f.rdbuf();
  auto doc = json::parse(ss.str());
  if (!doc) return fail(tokenizer_config_path + ": " + doc.error());
  const json::Value* tmpl = doc->find("chat_template");
  if (tmpl == nullptr || !tmpl->is_string()) return fail(tokenizer_config_path + ": no \"chat_template\" string field");
  return parse(tmpl->text());
}

Result<std::string> ChatTemplate::render(const std::vector<Message>& messages, bool add_generation_prompt) const {
  if (!root_) return fail("render: this ChatTemplate was default-constructed; call parse() or load() first");
  try {
    Env env;
    env.messages = &messages;
    env.add_generation_prompt = add_generation_prompt;
    return render_nodes(*root_, env);
  } catch (const std::exception& e) {
    return fail(e.what());
  }
}

}  // namespace llmi::chat
