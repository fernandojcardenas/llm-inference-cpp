// llmi-perplexity: windowed perplexity of a model on a fixed token sequence,
// used to measure M5's quantization quality loss (docs/quantization.md).
//
//   llmi-perplexity MODEL_DIR --ids-file FILE [--ctx 512] [--quant f32|q8_0|q4_0] [--full-window]
//
// FILE holds one decimal token id per line (blank lines ignored), produced
// by tools/make_ppl_ids.py from real text (the WikiText-2-raw test split,
// the standard small perplexity benchmark) tokenized with the model's own
// Hugging Face tokenizer -- the exact same ids are reused across every
// --quant run, so quantization's effect on perplexity isn't confounded by a
// different tokenization (this engine's own tokenizer only supports
// SmolLM2's vocabulary; Qwen2.5's needs the reference tokenizer either way,
// see docs/tokenizer.md).
//
// The sequence is split into non-overlapping chunks of --ctx tokens, each
// scored with one forward() call (all_positions=true). By default, only the
// SECOND HALF of each chunk is scored (log P(token[t] | token[0..t-1]) for
// t in [ctx/2, ctx)) -- the same convention llama.cpp's own perplexity tool
// uses by default (see its tools/perplexity/perplexity.cpp: "calculate the
// perplexity over the last half of the window, so the model always has some
// context to predict the token"), which this project's own first attempt at
// this tool didn't match: scoring every position from t=1 (minimal or no
// left-context for early positions in each chunk) gave a notably higher,
// NOT-comparable number (independently confirmed correct for THAT metric
// against a Hugging Face transformers reference computing the identical
// quantity -- see docs/evidence/m5-perplexity.txt). --full-window restores
// that simpler, non-comparable-to-llama.cpp metric, kept only because it's
// a slightly more informative (if less standard) measurement of a model's
// worst case (no left-context at all). Overall perplexity = exp(-mean log P
// over every scored token).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "llmi/model/transformer.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

bool read_ids(const std::string& path, std::vector<llmi::TokenId>& out) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    std::stringstream ss(line);
    std::string part;
    while (std::getline(ss, part, ',')) {
      if (part.find_first_not_of(" \t\r\n") == std::string::npos) continue;
      out.push_back(static_cast<llmi::TokenId>(std::stol(part)));
    }
  }
  return !out.empty();
}

bool parse_quant_type(const std::string& s, llmi::quant::Type& out) {
  if (s == "f32") {
    out = llmi::quant::Type::F32;
  } else if (s == "q8_0") {
    out = llmi::quant::Type::Q8_0;
  } else if (s == "q4_0") {
    out = llmi::quant::Type::Q4_0;
  } else {
    return false;
  }
  return true;
}

int run(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: llmi-perplexity MODEL_DIR --ids-file FILE [--ctx N] [--quant f32|q8_0|q4_0] "
                 "[--full-window]\n");
    return 2;
  }
  const std::string dir = argv[1];
  std::string ids_file;
  std::size_t ctx = 512;
  auto quant_type = llmi::quant::Type::F32;
  bool full_window = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a == "--ids-file" && has_value) {
      ids_file = argv[++i];
    } else if (a == "--ctx" && has_value) {
      ctx = std::stoul(argv[++i]);
    } else if (a == "--quant" && has_value) {
      if (!parse_quant_type(argv[++i], quant_type)) {
        std::fprintf(stderr, "error: --quant takes f32, q8_0 or q4_0\n");
        return 2;
      }
    } else if (a == "--full-window") {
      full_window = true;
    } else {
      std::fprintf(stderr, "error: unknown or incomplete argument %s\n", a.c_str());
      return 2;
    }
  }
  if (ids_file.empty()) {
    std::fprintf(stderr, "error: --ids-file is required\n");
    return 2;
  }

  std::vector<llmi::TokenId> ids;
  if (!read_ids(ids_file, ids)) {
    std::fprintf(stderr, "error: cannot read token ids from %s\n", ids_file.c_str());
    return 1;
  }

  const auto t_load = Clock::now();
  auto model = llmi::Model::load(dir);
  if (!model) {
    std::fprintf(stderr, "error: %s\n", model.error().c_str());
    return 1;
  }
  auto tf = llmi::Transformer::load(model.value(), quant_type);
  if (!tf) {
    std::fprintf(stderr, "error: %s\n", tf.error().c_str());
    return 1;
  }
  const double load_s = seconds_since(t_load);
  ctx = std::min(ctx, static_cast<std::size_t>(tf->config().max_position_embeddings));

  const std::size_t V = tf->config().vocab_size;
  double total_nll = 0.0;  // sum of -log P(actual next token), natural log
  std::size_t scored = 0;

  // Positions before `first` are still fed to the model (so later positions
  // get their real left-context) but aren't scored -- matching llama.cpp's
  // default (see the file comment above). --full-window scores from t=0
  // (well, t=1: the very first token of a chunk is never scored either way,
  // since it has no next-token prediction preceding it within the chunk).
  const std::size_t first = full_window ? 0 : ctx / 2;

  const auto t_eval = Clock::now();
  for (std::size_t start = 0; start + 1 < ids.size(); start += ctx) {
    const std::size_t end = std::min(start + ctx, ids.size());
    const std::vector<llmi::TokenId> chunk(ids.begin() + static_cast<std::ptrdiff_t>(start),
                                           ids.begin() + static_cast<std::ptrdiff_t>(end));
    auto logits = tf->forward(chunk, /*all_positions=*/true);
    if (!logits) {
      std::fprintf(stderr, "error: %s\n", logits.error().c_str());
      return 1;
    }
    // Position t's logits (row t) predict chunk[t+1]; the chunk's last
    // position has no "next token" within this chunk and isn't scored.
    const std::size_t t_start = std::min(first, !chunk.empty() ? chunk.size() - 1 : 0);
    for (std::size_t t = t_start; t + 1 < chunk.size(); ++t) {
      const float* row = logits->data() + t * V;
      const float mx = *std::max_element(row, row + V);
      double sum_exp = 0.0;
      for (std::size_t v = 0; v < V; ++v) sum_exp += std::exp(static_cast<double>(row[v] - mx));
      const double log_sum_exp = std::log(sum_exp) + mx;
      const auto next = static_cast<std::size_t>(chunk[t + 1]);
      const double log_p = static_cast<double>(row[next]) - log_sum_exp;
      total_nll -= log_p;
      ++scored;
    }
  }
  const double eval_s = seconds_since(t_eval);
  if (scored == 0) {
    std::fprintf(stderr, "error: fewer than 2 token ids after chunking; nothing to score\n");
    return 1;
  }
  const double mean_nll = total_nll / static_cast<double>(scored);
  const double perplexity = std::exp(mean_nll);

  std::printf(
      "{\"tokens\":%zu,\"scored_tokens\":%zu,\"ctx\":%zu,\"scored_from\":%zu,\"perplexity\":%.4f,\"mean_nll\":%.6f,"
      "\"weight_bytes\":%zu,\"load_seconds\":%.3f,\"eval_seconds\":%.3f}\n",
      ids.size(), scored, ctx, first, perplexity, mean_nll, tf->weight_bytes(), load_s, eval_s);
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
