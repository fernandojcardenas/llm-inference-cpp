// llmi-generate: greedy text generation with a model directory.
//
//   llmi-generate MODEL_DIR --prompt "The capital of France is" [--max-new 32]
//   llmi-generate MODEL_DIR --ids 504,3575,282 --max-new 16 --ignore-eos --json
//   llmi-generate MODEL_DIR --ids 504,3575,282 --dump trace.bin
//   llmi-generate MODEL_DIR --prompt "..." --max-new 32 --kv-cache
//   llmi-generate MODEL_DIR --prompt "..." --max-new 32 --kv-cache --quant q8_0
//   llmi-generate MODEL_DIR --prompt "..." --max-new 64 --json-schema schema.json
//   llmi-generate MODEL_DIR --prompt "..." --max-new 64 --grammar answer.gbnf
//   llmi-generate MODEL_DIR --prompt "..." --json-schema schema.json --temperature 0.8 --seed 1
//
// Constrained decoding (--grammar / --json-schema, ADR 0009) always uses the
// KV cache; it is greedy unless --temperature is given.
//
// --prompt needs a tokenizer.json the engine supports (SmolLM2's); --ids
// works with any supported model. --json prints the token ids and timing as
// one JSON object. --dump runs one forward pass over the prompt and writes
// every intermediate state and all logits (format: a JSON header line, then
// float32 little-endian arrays) for tools/crosscheck_forward.py. --kv-cache
// uses generate_greedy_cached (M3) instead of generate_greedy's no-cache
// baseline (M2); both must produce the same tokens (tests/transformer_test.cpp),
// so this flag only changes speed, used to measure M3's speed-up. --quant
// (f32, the default; q8_0; q4_0) block-quantizes every matmul weight at
// load time (M5, docs/quantization.md) -- --json's "weight_bytes" field
// reports this run's weight storage for M5's memory table.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "llmi/grammar/constraint.hpp"
#include "llmi/grammar/json_schema.hpp"
#include "llmi/model/transformer.hpp"
#include "llmi/util/json.hpp"
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

bool read_file(const std::string& path, std::string& out) {
  const std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream ss;
  ss << in.rdbuf();
  out = ss.str();
  return true;
}

// Loads --grammar (GBNF) or --json-schema (converted to GBNF), ADR 0009.
std::shared_ptr<const llmi::grammar::Grammar> load_grammar(const std::string& grammar_path,
                                                           const std::string& schema_path) {
  std::string text;
  const std::string& path = grammar_path.empty() ? schema_path : grammar_path;
  if (!read_file(path, text)) {
    std::fprintf(stderr, "error: cannot read %s\n", path.c_str());
    return nullptr;
  }
  if (!schema_path.empty()) {
    auto schema = llmi::json::parse(text);
    if (!schema) {
      std::fprintf(stderr, "error: %s: %s\n", path.c_str(), schema.error().c_str());
      return nullptr;
    }
    auto gbnf = llmi::grammar::json_schema_to_gbnf(*schema);
    if (!gbnf) {
      std::fprintf(stderr, "error: %s\n", gbnf.error().c_str());
      return nullptr;
    }
    text = std::move(gbnf.value());
  }
  auto g = llmi::grammar::Grammar::parse(text);
  if (!g) {
    std::fprintf(stderr, "error: %s: %s\n", path.c_str(), g.error().c_str());
    return nullptr;
  }
  return std::make_shared<const llmi::grammar::Grammar>(std::move(g.value()));
}

// Greedy generation with the KV cache, constrained to `grammar`. Prints the
// text and how long deciding which tokens were allowed took.
int generate_constrained(const llmi::Transformer& tf, const llmi::Tokenizer& tok,
                         const std::shared_ptr<const llmi::grammar::Grammar>& grammar,
                         const std::vector<llmi::TokenId>& ids, std::size_t max_new, bool as_json, double load_s,
                         const llmi::SamplingConfig& sampling, std::uint64_t seed) {
  const auto t_vocab = Clock::now();
  auto vocab = std::make_shared<const llmi::grammar::Vocabulary>(tok);
  const double vocab_s = seconds_since(t_vocab);
  const auto& eos_cfg = tf.config().eos_token_ids;
  llmi::grammar::Constraint constraint(grammar, vocab, std::vector<llmi::TokenId>(eos_cfg.begin(), eos_cfg.end()));
  std::mt19937_64 rng(seed);

  const auto t_gen = Clock::now();
  auto cache = tf.new_cache(ids.size() + max_new + 1);
  if (!cache) {
    std::fprintf(stderr, "error: %s\n", cache.error().c_str());
    return 1;
  }
  auto logits = tf.forward_cached(ids, *cache);
  std::vector<llmi::TokenId> out;
  bool hit_eos = false;
  for (std::size_t step = 0; step < max_new && logits; ++step) {
    auto next = constraint.pick(*logits, sampling, rng);
    if (!next) {
      std::fprintf(stderr, "error: %s\n", next.error().c_str());
      return 1;
    }
    if (!constraint.accept(*next)) {
      std::fprintf(stderr, "error: the grammar rejected its own pick\n");
      return 1;
    }
    if (std::find(eos_cfg.begin(), eos_cfg.end(), *next) != eos_cfg.end()) {
      hit_eos = true;
      break;
    }
    out.push_back(*next);
    if (constraint.finished()) break;
    logits = tf.forward_cached({*next}, *cache);
  }
  if (!logits) {
    std::fprintf(stderr, "error: %s\n", logits.error().c_str());
    return 1;
  }
  const double gen_s = seconds_since(t_gen);
  auto text = tok.decode(out);
  const auto& st = constraint.stats();
  if (as_json) {
    std::string quoted;
    llmi::json::append_quoted(quoted, text ? *text : "");
    std::printf(
        "{\"text\":%s,\"complete\":%s,\"generated_tokens\":%zu,\"generate_seconds\":%.3f,\"mask_seconds\":%.4f,"
        "\"full_masks\":%zu,\"trie_nodes_visited\":%zu,\"vocab_seconds\":%.3f,\"trie_nodes\":%zu}\n",
        quoted.c_str(), constraint.complete() ? "true" : "false", out.size(), gen_s, st.mask_seconds, st.full_masks,
        st.trie_nodes_visited, vocab_s, vocab->trie.node_count());
    return 0;
  }
  std::printf("%s\n", text ? text->c_str() : "");
  std::printf(
      "[%zu prompt tokens, %zu generated in %.2f s; grammar %s%s; deciding allowed tokens took %.3f s (%zu of %zu "
      "picks needed a full vocabulary check); vocabulary trie %zu nodes built in %.2f s; load %.2f s]\n",
      ids.size(), out.size(), gen_s, constraint.complete() ? "complete" : "INCOMPLETE",
      hit_eos ? ", ended with end-of-text" : "", st.mask_seconds, st.full_masks, st.picks,
      vocab->trie.node_count(), vocab_s, load_s);
  return constraint.complete() ? 0 : 3;
}

int run(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: llmi-generate MODEL_DIR (--prompt TEXT | --ids N,N,...) [--max-new N] [--ignore-eos] [--json] "
                 "[--dump FILE] [--kv-cache] [--quant f32|q8_0|q4_0] [--grammar GBNF_FILE | --json-schema SCHEMA_FILE [--temperature T] [--top-p P] [--seed N]]\n");
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
  std::string grammar_path;
  std::string schema_path;
  llmi::SamplingConfig sampling{0.0F, 0, 1.0F};  // constrained mode only; greedy unless --temperature
  std::uint64_t seed = 0;
  auto quant_type = llmi::quant::Type::F32;
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
    } else if (a == "--quant" && has_value) {
      if (!parse_quant_type(argv[++i], quant_type)) {
        std::fprintf(stderr, "error: --quant takes f32, q8_0 or q4_0\n");
        return 2;
      }
    } else if (a == "--grammar" && has_value) {
      grammar_path = argv[++i];
    } else if (a == "--json-schema" && has_value) {
      schema_path = argv[++i];
    } else if (a == "--temperature" && has_value) {
      sampling.temperature = std::stof(argv[++i]);
    } else if (a == "--top-p" && has_value) {
      sampling.top_p = std::stof(argv[++i]);
    } else if (a == "--seed" && has_value) {
      seed = std::stoull(argv[++i]);
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
  auto tf = llmi::Transformer::load(model.value(), quant_type);
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

  if (!grammar_path.empty() || !schema_path.empty()) {
    if (!grammar_path.empty() && !schema_path.empty()) {
      std::fprintf(stderr, "error: give --grammar or --json-schema, not both\n");
      return 2;
    }
    if (!tok) {
      std::fprintf(stderr, "error: constrained decoding needs the model's tokenizer.json\n");
      return 1;
    }
    auto grammar = load_grammar(grammar_path, schema_path);
    if (!grammar) return 1;
    return generate_constrained(tf.value(), *tok, grammar, ids, max_new, as_json, load_s, sampling, seed);
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
    std::printf(
        "{\"prompt_ids\":[%s],\"generated_ids\":[%s],\"load_seconds\":%.3f,\"generate_seconds\":%.3f,"
        "\"weight_bytes\":%zu}\n",
        join(ids).c_str(), join(out.value()).c_str(), load_s, gen_s, tf->weight_bytes());
    return 0;
  }
  if (tok) {
    auto text = tok->decode(out.value());
    std::printf("%s%s\n", prompt.c_str(), text ? text->c_str() : "");
  } else {
    std::printf("generated ids: %s\n", join(out.value()).c_str());
  }
  std::printf("[%zu prompt tokens, %zu generated in %.2f s (%.2f tokens/s, %s); load %.2f s; %zu weight bytes]\n",
              ids.size(), out->size(), gen_s, static_cast<double>(out->size()) / gen_s,
              use_kv_cache ? "KV cache" : "no KV cache", load_s, tf->weight_bytes());
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
