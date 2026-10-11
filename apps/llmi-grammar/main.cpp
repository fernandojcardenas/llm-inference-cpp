// llmi-grammar: check a grammar, convert a JSON Schema, test text against
// either (M8, ADR 0009).
//
//   llmi-grammar --grammar FILE                 parse and summarize a GBNF grammar
//   llmi-grammar --schema FILE                  print the grammar for a JSON Schema
//   llmi-grammar (--grammar|--schema) FILE --jsonl < in.jsonl
//
// Batch mode reads one JSON string per line and writes one character per
// line: 'C' if the string is a complete sentence of the grammar, 'P' if it
// is not but can still be extended into one (a viable prefix), '-' if
// neither. tools/crosscheck_grammar.py compares this against an independent
// recognizer.

#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "llmi/grammar/grammar.hpp"
#include "llmi/grammar/json_schema.hpp"
#include "llmi/grammar/matcher.hpp"
#include "llmi/util/json.hpp"

namespace {

bool read_file(const char* path, std::string& out) {
  const std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

int run(int argc, char** argv) {
  if (argc < 3 || (std::strcmp(argv[1], "--grammar") != 0 && std::strcmp(argv[1], "--schema") != 0)) {
    std::fprintf(stderr, "usage: llmi-grammar (--grammar GBNF_FILE | --schema JSON_SCHEMA_FILE) [--jsonl]\n");
    return 2;
  }
  const bool is_schema = std::strcmp(argv[1], "--schema") == 0;
  const bool batch = argc > 3 && std::strcmp(argv[3], "--jsonl") == 0;

  std::string text;
  if (!read_file(argv[2], text)) {
    std::fprintf(stderr, "error: cannot read %s\n", argv[2]);
    return 1;
  }
  if (is_schema) {
    auto schema = llmi::json::parse(text);
    if (!schema) {
      std::fprintf(stderr, "error: %s\n", schema.error().c_str());
      return 1;
    }
    auto gbnf = llmi::grammar::json_schema_to_gbnf(*schema);
    if (!gbnf) {
      std::fprintf(stderr, "error: %s\n", gbnf.error().c_str());
      return 1;
    }
    text = std::move(gbnf.value());
    if (!batch) std::fputs(text.c_str(), stdout);
  }

  auto g = llmi::grammar::Grammar::parse(text);
  if (!g) {
    std::fprintf(stderr, "error: %s\n", g.error().c_str());
    return 1;
  }
  const auto grammar = std::make_shared<const llmi::grammar::Grammar>(std::move(g.value()));
  if (!batch) {
    if (!is_schema) {
      std::printf("ok: %zu rules, %zu symbols after desugaring\n", grammar->rules().size(), grammar->symbol_count());
    }
    return 0;
  }

  std::ios::sync_with_stdio(false);
  std::string line;
  std::size_t n = 0;
  while (std::getline(std::cin, line)) {
    ++n;
    auto v = llmi::json::parse(line);
    if (!v || !v->is_string()) {
      std::fprintf(stderr, "error: line %zu is not a JSON string\n", n);
      return 1;
    }
    llmi::grammar::Matcher m(grammar);
    const bool ok = m.accept(v->text());
    char verdict = '-';
    if (ok) verdict = m.complete() ? 'C' : 'P';
    std::cout << verdict << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
