#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "llmi/grammar/grammar.hpp"
#include "llmi/grammar/json_schema.hpp"
#include "llmi/grammar/matcher.hpp"

using llmi::grammar::Grammar;
using llmi::grammar::Matcher;
using llmi::grammar::TokenTrie;

namespace {

std::shared_ptr<const Grammar> compile(const std::string& text) {
  auto g = Grammar::parse(text);
  EXPECT_TRUE(g.ok()) << (g.ok() ? "" : g.error());
  if (!g.ok()) return nullptr;
  return std::make_shared<const Grammar>(std::move(g.value()));
}

// The whole of `s` is a sentence of the grammar.
bool matches(const std::shared_ptr<const Grammar>& g, const std::string& s) {
  Matcher m(g);
  return m.accept(s) && m.complete();
}

// `s` can still be extended into a sentence.
bool prefix_ok(const std::shared_ptr<const Grammar>& g, const std::string& s) {
  Matcher m(g);
  return m.accept(s);
}

std::string parse_error(const std::string& text) {
  auto g = Grammar::parse(text);
  return g.ok() ? std::string("(parsed)") : g.error();
}

}  // namespace

TEST(GrammarParse, RejectsWhatTheMatcherCouldNotRunSafely) {
  EXPECT_NE(parse_error("a ::= \"x\"").find("no 'root' rule"), std::string::npos);
  EXPECT_NE(parse_error("root ::= missing").find("never defined"), std::string::npos);
  EXPECT_NE(parse_error("root ::= root \"x\" | \"y\"").find("left-recursive"), std::string::npos);
  // Indirect, and hidden behind a rule that can match nothing.
  EXPECT_NE(parse_error("root ::= a\na ::= b \"x\"\nb ::= root | \"y\"").find("left-recursive"), std::string::npos);
  EXPECT_NE(parse_error("root ::= e root \"x\" | \"y\"\ne ::= \"\" | \"z\"").find("left-recursive"), std::string::npos);
  EXPECT_NE(parse_error("root ::= \"a\"\nroot ::= \"b\"").find("defined twice"), std::string::npos);
  EXPECT_NE(parse_error("root ::= \"a\\q\"").find("unknown escape"), std::string::npos);
  EXPECT_NE(parse_error("root ::= \"abc").find("unterminated"), std::string::npos);
  EXPECT_NE(parse_error("root ::= [a-z").find("unterminated"), std::string::npos);
  EXPECT_NE(parse_error("root ::= [z-a]").find("backwards"), std::string::npos);
  EXPECT_NE(parse_error("root ::= []").find("matches nothing"), std::string::npos);
  EXPECT_NE(parse_error("root ::= \"a\"{2001}").find("repetition count"), std::string::npos);
  EXPECT_NE(parse_error("root ::= \"a\"{3,2}").find("n < m"), std::string::npos);
  EXPECT_NE(parse_error("root ::= (\"a\"").find("expected ')'"), std::string::npos);
  EXPECT_NE(parse_error("root ::= \"\\uD800\"").find("not a valid code point"), std::string::npos);
  EXPECT_NE(parse_error("").find("empty"), std::string::npos);
  EXPECT_NE(parse_error(std::string("root ::= \"\xff\"")).find("UTF-8"), std::string::npos);
}

TEST(GrammarParse, EnforcesItsLimits) {
  llmi::grammar::Limits small;
  small.max_grammar_bytes = 16;
  EXPECT_FALSE(Grammar::parse("root ::= \"aaaaaaaaaaaaaaaa\"", small).ok());

  llmi::grammar::Limits few;
  few.max_symbols = 50;
  EXPECT_FALSE(Grammar::parse("root ::= \"a\"{100}", few).ok());

  std::string deep = "root ::= ";
  for (int i = 0; i < 70; ++i) deep += "(";
  deep += "\"a\"";
  for (int i = 0; i < 70; ++i) deep += ")";
  EXPECT_NE(parse_error(deep).find("nested too deeply"), std::string::npos);
}

TEST(GrammarMatch, LiteralsAlternativesAndClasses) {
  auto g = compile(R"(root ::= "yes" | "no" | [0-9] [a-f]?)");
  EXPECT_TRUE(matches(g, "yes"));
  EXPECT_TRUE(matches(g, "no"));
  EXPECT_TRUE(matches(g, "7"));
  EXPECT_TRUE(matches(g, "7c"));
  EXPECT_FALSE(matches(g, "ye"));     // a prefix, not a sentence
  EXPECT_TRUE(prefix_ok(g, "ye"));
  EXPECT_FALSE(prefix_ok(g, "yez"));
  EXPECT_FALSE(prefix_ok(g, "7g"));
  EXPECT_FALSE(prefix_ok(g, "yess"));
}

TEST(GrammarMatch, NegatedClassesAndAnyCharacter) {
  auto g = compile(R"(root ::= "\"" [^"\\]* "\"" .)");
  EXPECT_TRUE(matches(g, "\"hello world\"!"));
  EXPECT_TRUE(matches(g, "\"\"x"));
  EXPECT_FALSE(prefix_ok(g, "\"a\\"));
  EXPECT_FALSE(matches(g, "\"abc\""));  // '.' needs one more character
}

TEST(GrammarMatch, CountedRepetition) {
  auto g = compile(R"(root ::= "a"{2,4} "b"{3} "c"{1,})");
  EXPECT_FALSE(prefix_ok(g, "ab"));
  EXPECT_TRUE(matches(g, "aabbbc"));
  EXPECT_TRUE(matches(g, "aaaabbbccc"));
  EXPECT_FALSE(prefix_ok(g, "aaaaa"));
  EXPECT_FALSE(prefix_ok(g, "aabbbb"));
  EXPECT_FALSE(matches(g, "aabbb"));
  // A repeated multi-character literal and a repeated group.
  auto h = compile(R"(root ::= "ab"{2} ("x" | "yz")+)");
  EXPECT_TRUE(matches(h, "ababyzx"));
  EXPECT_FALSE(prefix_ok(h, "abx"));
}

TEST(GrammarMatch, RightRecursionAndNesting) {
  auto g = compile("root ::= \"(\" root \")\" root | \"\"");
  EXPECT_TRUE(matches(g, ""));
  EXPECT_TRUE(matches(g, "(()())()"));
  EXPECT_FALSE(matches(g, "(()"));
  EXPECT_TRUE(prefix_ok(g, "(()"));
  EXPECT_FALSE(prefix_ok(g, "())"));
}

TEST(GrammarMatch, RulesMayBeUsedBeforeDefinitionAndSpanLines) {
  auto g = compile(R"(
    # a comment
    root ::= greeting " " name
    greeting ::= "hi"
               | "hello"
    name ::= [A-Z] [a-z]+
  )");
  EXPECT_TRUE(matches(g, "hello Ada"));
  EXPECT_TRUE(matches(g, "hi Bo"));
  EXPECT_FALSE(prefix_ok(g, "hi bo"));
}

TEST(GrammarMatch, MultiByteCharactersAcrossSplitBytes) {
  auto g = compile("root ::= \"caf\" [\\u00e0-\\u00ff] \"!\"");
  EXPECT_TRUE(matches(g, "caf\xc3\xa9!"));  // é
  Matcher m(g);
  ASSERT_TRUE(m.accept("caf"));
  ASSERT_TRUE(m.accept("\xc3"));  // half of é: some allowed character starts with this byte
  EXPECT_FALSE(m.complete());
  ASSERT_TRUE(m.accept("\xa9"));
  ASSERT_TRUE(m.accept("!"));
  EXPECT_TRUE(m.complete());

  Matcher n(g);
  ASSERT_TRUE(n.accept("caf"));
  EXPECT_FALSE(n.can_accept("\xe2"));    // a 3-byte character can't be in U+00E0..U+00FF
  EXPECT_FALSE(n.can_accept("\xc2"));    // U+0080..U+00BF: below the class
  EXPECT_FALSE(n.can_accept("\xc0\xa9"));  // overlong encoding
  EXPECT_FALSE(n.can_accept("\xa9"));    // a lone continuation byte
}

TEST(GrammarMatch, RejectionLeavesTheStateUnchanged) {
  auto g = compile(R"(root ::= "ab" "c")");
  Matcher m(g);
  ASSERT_TRUE(m.accept("a"));
  EXPECT_FALSE(m.accept("bx"));  // 'b' alone was fine, but the pair is not
  EXPECT_TRUE(m.accept("bc"));
  EXPECT_TRUE(m.complete());
  EXPECT_FALSE(m.can_accept("c"));
}

TEST(GrammarMatch, AmbiguityIsDeduplicatedAndBounded) {
  // Every split of the a's is a parse; without deduplication the number of
  // stacks would grow with the input.
  auto g = compile(R"(root ::= x*  x ::= "a" | "aa" | "aaa")");
  Matcher m(g);
  ASSERT_TRUE(m.accept(std::string(500, 'a')));
  EXPECT_TRUE(m.complete());

  // A grammar whose stack count genuinely explodes fails closed.
  auto h = compile(R"(root ::= ("a" b)*  b ::= "" | "a" b | "a")");
  Matcher::Limits tiny;
  tiny.max_stacks = 4;
  Matcher small(h, tiny);
  EXPECT_FALSE(small.accept(std::string(64, 'a')));
}

TEST(GrammarMatch, TokenMaskEqualsCheckingEachTokenOnItsOwn) {
  auto g = compile(R"(root ::= "{" [ ]? "\"" [a-z]+ "\"" [ ]? "}")");
  const std::vector<std::string> vocab = {"",     "{",     "{\"",  "\"",  "a",   "abc", "\"}",  " }",   "}",
                                          "{ \"", "x\"",   "\xc3", "1",   "ab",  "{{",  "\" }", "\"\"", "zz\"}"};
  const TokenTrie trie(vocab);
  std::vector<std::uint8_t> mask;
  for (const std::string& prefix : {std::string(), std::string("{"), std::string("{\"ab"), std::string("{\"ab\"")}) {
    Matcher m(g);
    ASSERT_TRUE(m.accept(prefix)) << prefix;
    m.allowed_tokens(trie, mask);
    ASSERT_EQ(mask.size(), vocab.size());
    for (std::size_t id = 0; id < vocab.size(); ++id) {
      const bool expect = !vocab[id].empty() && m.can_accept(vocab[id]);
      EXPECT_EQ(mask[id] == 1, expect) << "prefix '" << prefix << "' token '" << vocab[id] << "'";
    }
  }
}

TEST(GrammarMatch, MemoizedMaskEqualsTheDirectWalk) {
  // allowed_tokens() memoizes transitions; allowed_tokens_direct() is the
  // original per-node walk. They must agree byte for byte at every state.
  // The vocabulary gives every printable ASCII byte its own token, so a
  // free-form string allows almost all of them -- the case the memo collapses
  // and the one most likely to expose a divergence -- plus structural and
  // multi-byte tokens.
  std::vector<std::string> vocab;
  vocab.emplace_back();  // id 0: no bytes, never allowed
  for (int c = 0x20; c < 0x7f; ++c) vocab.emplace_back(1, static_cast<char>(c));
  for (const char* t : {"{\"", "\":", "\",", "\"}", "ab", "abc", "123", "true", "false", "null", ", ", ": ",
                        "\xc3\xa9", "\xc3", "\xa9", "\xe2\x9c\x93"}) {
    vocab.emplace_back(t);
  }
  const TokenTrie trie(vocab);

  std::vector<std::string> grammars = {
      "root ::= \"\\\"\" char* \"\\\"\"\nchar ::= [^\"]\n",  // a free-form JSON string
      "root ::= \"[\" ws item (ws \",\" ws item)* ws \"]\"\nitem ::= [a-z]+\nws ::= [ \\t]*\n",
      "root ::= \"{\" ws \"\\\"\" [a-z]+ \"\\\"\" ws \":\" ws val ws \"}\"\n"
      "val ::= \"\\\"\" [^\"]* \"\\\"\" | [0-9]+ | \"true\" | \"false\" | \"null\"\nws ::= [ \\t\\n]*\n",
      llmi::grammar::json_object_grammar(),
  };

  for (const std::string& text : grammars) {
    const auto g = compile(text);
    ASSERT_NE(g, nullptr);
    Matcher m(g);
    std::vector<std::uint8_t> memo;
    std::vector<std::uint8_t> direct;
    for (int step = 0; step < 60; ++step) {
      m.allowed_tokens(trie, memo);
      m.allowed_tokens_direct(trie, direct);
      ASSERT_EQ(memo, direct) << "grammar:\n" << text << "step " << step;
      int picked = -1;  // advance by the lowest-id allowed, non-empty token
      for (std::size_t id = 0; id < memo.size(); ++id) {
        if (memo[id] != 0 && !vocab[id].empty()) {
          picked = static_cast<int>(id);
          break;
        }
      }
      if (picked < 0) break;
      if (m.complete() && step >= 4) break;
      ASSERT_TRUE(m.accept(vocab[static_cast<std::size_t>(picked)]));
      if (m.complete() && !m.has_continuation()) break;
    }
  }
}

TEST(GrammarMatch, TrieHandlesDuplicateAndPrefixTokens) {
  const std::vector<std::string> vocab = {"ab", "a", "ab", "abc", "b"};
  const TokenTrie trie(vocab);
  EXPECT_EQ(trie.node_count(), 5U);  // root, a, ab, abc, b
  auto g = compile(R"(root ::= "ab")");
  Matcher m(g);
  std::vector<std::uint8_t> mask;
  m.allowed_tokens(trie, mask);
  EXPECT_EQ(mask, (std::vector<std::uint8_t>{1, 1, 1, 0, 0}));
}
