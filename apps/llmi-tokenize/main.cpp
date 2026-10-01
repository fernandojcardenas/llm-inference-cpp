// llmi-tokenize: encode text with a tokenizer.json.
//
//   llmi-tokenize TOKENIZER_JSON "Hello, world"     token ids and pieces
//   llmi-tokenize TOKENIZER_JSON --jsonl < in.jsonl  batch mode
//
// Batch mode reads one JSON string per line and writes one JSON array of
// token ids per line. Every line is also decoded again and compared with the
// input; a mismatch is reported and makes the exit status 1. Lines that lose
// bytes the vocabulary has no token for (the reference drops them too) are
// counted separately and skip the round-trip check.

#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "llmi/tokenizer/tokenizer.hpp"
#include "llmi/util/json.hpp"

namespace {

int run(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: llmi-tokenize TOKENIZER_JSON (TEXT | --jsonl)\n");
    return 2;
  }
  auto tok = llmi::Tokenizer::load(argv[1]);
  if (!tok) {
    std::fprintf(stderr, "error: %s\n", tok.error().c_str());
    return 1;
  }

  if (std::strcmp(argv[2], "--jsonl") != 0) {
    std::size_t dropped = 0;
    auto ids = tok->encode(argv[2], true, &dropped);
    if (!ids) {
      std::fprintf(stderr, "error: %s\n", ids.error().c_str());
      return 1;
    }
    for (const auto id : ids.value()) {
      std::string piece;
      llmi::json::append_quoted(piece, tok->token(id));
      std::printf("%6d  %s\n", id, piece.c_str());
    }
    std::printf("%zu tokens\n", ids->size());
    if (dropped != 0) std::printf("warning: %zu bytes have no token in this vocabulary and were dropped\n", dropped);
    return 0;
  }

  std::ios::sync_with_stdio(false);
  std::string line;
  std::size_t lines = 0;
  std::size_t tokens = 0;
  std::size_t mismatches = 0;
  std::size_t lossy_lines = 0;
  std::size_t dropped_bytes = 0;
  std::string out;
  while (std::getline(std::cin, line)) {
    ++lines;
    auto v = llmi::json::parse(line);
    if (!v || !v->is_string()) {
      std::fprintf(stderr, "line %zu: not a JSON string\n", lines);
      return 1;
    }
    std::size_t dropped = 0;
    auto ids = tok->encode(v->text(), true, &dropped);
    if (!ids) {
      std::fprintf(stderr, "line %zu: %s\n", lines, ids.error().c_str());
      return 1;
    }
    tokens += ids->size();
    if (dropped != 0) {
      ++lossy_lines;
      dropped_bytes += dropped;
    }
    auto back = tok->decode(ids.value());
    if (dropped == 0 && (!back || back.value() != v->text())) {
      ++mismatches;
      std::fprintf(stderr, "line %zu: decode(encode(text)) != text\n", lines);
    }
    out.clear();
    out.push_back('[');
    for (std::size_t i = 0; i < ids->size(); ++i) {
      if (i != 0) out.push_back(',');
      out += std::to_string(ids.value()[i]);
    }
    out += "]\n";
    std::fputs(out.c_str(), stdout);
  }
  std::fprintf(stderr, "lines %zu, tokens %zu, round-trip mismatches %zu, lines with dropped bytes %zu (%zu bytes)\n",
               lines, tokens, mismatches, lossy_lines, dropped_bytes);
  return mismatches == 0 ? 0 : 1;
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
