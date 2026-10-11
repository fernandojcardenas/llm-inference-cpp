// Arbitrary bytes as a grammar and as text matched against it -- grammars
// reach this engine from HTTP requests (response_format, ADR 0009), so the
// parser and the matcher are an untrusted-input boundary like every loader.
//
// Input layout: grammar text, a 0x00 byte, then text to match. Without a
// 0x00 byte the whole input is also tried as a JSON Schema.
//
// Invariants: no crash or sanitizer report; the matcher never accepts a byte
// that can_accept() refused; every token the mask allows is accepted on its
// own; and any schema the converter accepts yields grammar text the parser
// accepts.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "llmi/grammar/grammar.hpp"
#include "llmi/grammar/json_schema.hpp"
#include "llmi/grammar/matcher.hpp"
#include "llmi/util/json.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view in(reinterpret_cast<const char*>(data), size);
  const std::size_t split = in.find('\0');
  const std::string_view grammar_text = in.substr(0, split);
  const std::string_view text = split == std::string_view::npos ? std::string_view() : in.substr(split + 1);

  if (split == std::string_view::npos) {
    if (auto schema = llmi::json::parse(in)) {
      if (auto gbnf = llmi::grammar::json_schema_to_gbnf(*schema)) {
        if (!llmi::grammar::Grammar::parse(*gbnf).ok()) std::abort();
      }
    }
  }

  auto g = llmi::grammar::Grammar::parse(grammar_text);
  if (!g) return 0;
  llmi::grammar::Matcher::Limits limits;
  limits.max_stacks = 256;  // keep each run fast; the limits are what is being exercised
  limits.max_expansions = 20'000;
  llmi::grammar::Matcher m(std::make_shared<const llmi::grammar::Grammar>(std::move(g.value())), limits);

  static const std::vector<std::string> kVocab = {"a", "ab", "{", "\"", " ", "0", "12", "\xc3", "\xa9", "\xc3\xa9",
                                                  "}", "\n", "x\"", "true", "null", ",", ":", "[", "]", "\xe2\x82"};
  static const llmi::grammar::TokenTrie kTrie(kVocab);
  std::vector<std::uint8_t> mask;
  for (std::size_t i = 0; i < text.size() && i < 256; ++i) {
    const std::string_view byte = text.substr(i, 1);
    const bool can = m.can_accept(byte);
    const bool did = m.accept(byte);
    if (can != did) std::abort();
    if ((i % 16) == 0) {
      m.allowed_tokens(kTrie, mask);
      for (std::size_t id = 0; id < kVocab.size(); ++id) {
        if (mask[id] == 1 && !m.can_accept(kVocab[id])) std::abort();
      }
    }
    (void)m.complete();
  }
  return 0;
}
