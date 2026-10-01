// llmi-generate: greedy text generation with a model directory.
//
//   llmi-generate MODEL_DIR --prompt "The capital of France is" [--max-new 32]
//   llmi-generate MODEL_DIR --ids 504,3575,282 --max-new 16 --ignore-eos --json
//   llmi-generate MODEL_DIR --ids 504,3575,282 --dump trace.bin
//   llmi-generate MODEL_DIR --prompt "..." --max-new 32 --kv-cache
//
// --prompt needs a tokenizer.json the engine supports (SmolLM2's); --ids
// works with any supported model. --json prints the token ids and timing as
// one JSON object. --dump runs one forward pass over the prompt and writes
// every intermediate state and all logits (format: a JSON header line, then
// float32 little-endian arrays) for tools/crosscheck_forward.py. --kv-cache
// uses generate_greedy_cached (M3) instead of generate_greedy's no-cache
// baseline (M2); both must produce the same tokens (tests/transformer_test.cpp),
// so this flag only changes speed, used to measure M3's speed-up.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "llmi/model/transformer.hpp"
#include "llmi/tokenizer/tokenizer.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

bool parse_ids(const std::string& s, std::vector<llmi::TokenId>& out) {
  std::size_t i = 0;
  while (i < s.size()) {
    std::size_t j = s.find(',', i);
    if (j == std::string::npos) j = s.size();
    const std::string part = s.substr(i, j - i);
    if (part.empty() || part.find_first_not_of("0123456789") != std::string::npos || part.size() > 9) return false;
    out.push_back(static_cast<llmi::TokenId>(std::stol(part)));
    i = j + 1;
  }
  return !out.empty();
}

std::string join(const std::vector<llmi::TokenId>& ids) {
  std::string s;
  for (std::size_t i = 0; i < ids.size(); ++i) s += (i != 0 ? "," : "") + std::to_string(ids[i]);
  return s;
}

bool write_dump(const std::string& path, const llmi::ForwardTrace& trace, const std::vector<float>& logits,
                std::size_t tokens, std::size_t hidden, std::size_t vocab) {
  std::ofstream out(path, std::ios::binary);
  const std::string header = "{\"tokens\":" + std::to_string(tokens) + ",\"hidden\":" + std::to_string(hidden) +
                             ",\"vocab\":" + std::to_string(vocab) + ",\"states\":" + std::to_string(trace.states.size()) +
                             "}\n";
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  for (const auto& s : trace.states) {
    out.write(reinterpret_cast<const char*>(s.data()), static_cast<std::streamsize>(s.size() * sizeof(float)));
  }
  out.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
  return static_cast<bool>(out);
}

int run(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: llmi-generate MODEL_DIR (--prompt TEXT | --ids N,N,...) [--max-new N] [--ignore-eos] [--json] "
                 "[--dump FILE]\n");
    return 2;
  }
  const std::string dir = argv[1];
  std::string prompt;
  std::string dump;
  std::vector<llmi::TokenId> ids;
  std::size_t max_new = 32;
  bool ignore_eos = false;
  bool as_json = false;
  bool use_kv_cache = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const bool has_value = i + 1 < argc;
    if (a == "--prompt" && has_value) {
      prompt = argv[++i];
    } else if (a == "--ids" && has_value) {
      if (!parse_ids(argv[++i], ids)) {
        std::fprintf(stderr, "error: --ids takes comma-separated token ids\n");
        return 2;
      }
    } else if (a == "--max-new" && has_value) {
      max_new = std::stoul(argv[++i]);
    } else if (a == "--dump" && has_value) {
      dump = argv[++i];
    } else if (a == "--ignore-eos") {
      ignore_eos = true;
    } else if (a == "--json") {
      as_json = true;
    } else if (a == "--kv-cache") {
      use_kv_cache = true;
    } else {
      std::fprintf(stderr, "error: unknown or incomplete argument %s\n", a.c_str());
      return 2;
    }
  }

  const auto t_load = Clock::now();
  auto model = llmi::Model::load(dir);
  if (!model) {
    std::fprintf(stderr, "error: %s\n", model.error().c_str());
    return 1;
  }
  auto tf = llmi::Transformer::load(model.value());
  if (!tf) {
    std::fprintf(stderr, "error: %s\n", tf.error().c_str());
    return 1;
  }
  const double load_s = seconds_since(t_load);

  std::optional<llmi::Tokenizer> tok;
  if (auto t = llmi::Tokenizer::load(dir + "/tokenizer.json")) tok = std::move(t.value());
  if (!prompt.empty()) {
    if (!tok) {
      std::fprintf(stderr, "error: this model's tokenizer is not supported yet; pass --ids instead\n");
      return 1;
    }
    std::size_t dropped = 0;
    auto enc = tok->encode(prompt, true, &dropped);
    if (!enc) {
      std::fprintf(stderr, "error: %s\n", enc.error().c_str());
      return 1;
    }
    if (dropped != 0) std::fprintf(stderr, "warning: %zu prompt bytes have no token and were dropped\n", dropped);
    ids = enc.value();
  }
  if (ids.empty()) {
    std::fprintf(stderr, "error: give --prompt or --ids\n");
    return 2;
  }

  if (!dump.empty()) {
    llmi::ForwardTrace trace;
    auto logits = tf->forward(ids, /*all_positions=*/true, &trace);
    if (!logits) {
      std::fprintf(stderr, "error: %s\n", logits.error().c_str());
      return 1;
    }
    const auto& c = tf->config();
    if (!write_dump(dump, trace, logits.value(), ids.size(), c.hidden_size, c.vocab_size)) {
      std::fprintf(stderr, "error: cannot write %s\n", dump.c_str());
      return 1;
    }
    if (max_new == 0) return 0;
  }

  const auto t_gen = Clock::now();
  const std::vector<llmi::TokenId> none;
  const auto& stop_ids = ignore_eos ? none : tf->config().eos_token_ids;
  auto out = use_kv_cache ? tf->generate_greedy_cached(ids, max_new, stop_ids) : tf->generate_greedy(ids, max_new, stop_ids);
  if (!out) {
    std::fprintf(stderr, "error: %s\n", out.error().c_str());
    return 1;
  }
  const double gen_s = seconds_since(t_gen);

  if (as_json) {
    std::printf("{\"prompt_ids\":[%s],\"generated_ids\":[%s],\"load_seconds\":%.3f,\"generate_seconds\":%.3f}\n",
                join(ids).c_str(), join(out.value()).c_str(), load_s, gen_s);
    return 0;
  }
  if (tok) {
    auto text = tok->decode(out.value());
    std::printf("%s%s\n", prompt.c_str(), text ? text->c_str() : "");
  } else {
    std::printf("generated ids: %s\n", join(out.value()).c_str());
  }
  std::printf("[%zu prompt tokens, %zu generated in %.2f s (%.2f tokens/s, %s); load %.2f s]\n", ids.size(),
              out->size(), gen_s, static_cast<double>(out->size()) / gen_s, use_kv_cache ? "KV cache" : "no KV cache",
              load_s);
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
