// llmi-chat: an interactive chat CLI using a KV cache, sampling and the
// model's own chat template.
//
//   llmi-chat MODEL_DIR [--system TEXT] [--temperature F] [--top-k N]
//                        [--top-p F] [--seed N] [--max-new N]
//
// Reads lines from stdin as user turns, prints the assistant's reply as it
// is generated, and keeps the conversation (and the KV cache) across turns.
// MODEL_DIR needs a tokenizer_config.json with a chat_template (the engine's
// own tokenizer.json does not support Qwen2.5's tokenizer yet, so this app
// requires --ids-compatible models or a tokenizer the engine supports -- see
// docs/sampling.md for which models work end to end today).
//
// Across turns, the whole conversation is re-rendered through the chat
// template and re-tokenized; only the tokens beyond the longest common
// prefix with what is already cached are actually recomputed (ADR 0004).
// This is usually just the new turn, but a cache reset and full reprefill
// is the (rare, and still correct) fallback if retokenization ever disagrees
// with the cached prefix (a BPE merge spanning the old/new boundary).

#include <cstdio>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "llmi/chat/template.hpp"
#include "llmi/model/sampling.hpp"
#include "llmi/model/transformer.hpp"
#include "llmi/tokenizer/tokenizer.hpp"

namespace {

struct Args {
  std::string model_dir;
  std::string system;
  float temperature = 0.0F;  // default: greedy, for reproducible demos
  std::size_t top_k = 0;
  float top_p = 1.0F;
  std::uint64_t seed = 0;
  std::size_t max_new = 256;
};

bool parse_args(int argc, char** argv, Args* a) {
  if (argc < 2) return false;
  a->model_dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--system" && has_value) a->system = argv[++i];
    else if (arg == "--temperature" && has_value) a->temperature = std::stof(argv[++i]);
    else if (arg == "--top-k" && has_value) a->top_k = std::stoul(argv[++i]);
    else if (arg == "--top-p" && has_value) a->top_p = std::stof(argv[++i]);
    else if (arg == "--seed" && has_value) a->seed = std::stoull(argv[++i]);
    else if (arg == "--max-new" && has_value) a->max_new = std::stoul(argv[++i]);
    else return false;
  }
  return true;
}

// Index of the first position where a and b differ, or the shorter length.
std::size_t common_prefix_length(const std::vector<llmi::TokenId>& a, const std::vector<llmi::TokenId>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  std::size_t i = 0;
  while (i < n && a[i] == b[i]) ++i;
  return i;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, &args)) {
    std::fprintf(stderr,
                 "usage: %s MODEL_DIR [--system TEXT] [--temperature F] [--top-k N] [--top-p F] [--seed N] "
                 "[--max-new N]\n",
                 argv[0]);
    return 2;
  }

  auto model = llmi::Model::load(args.model_dir);
  if (!model) {
    std::fprintf(stderr, "error: %s\n", model.error().c_str());
    return 1;
  }
  auto transformer = llmi::Transformer::load(model.value());
  if (!transformer) {
    std::fprintf(stderr, "error: %s\n", transformer.error().c_str());
    return 1;
  }
  auto tokenizer = llmi::Tokenizer::load(args.model_dir + "/tokenizer.json");
  if (!tokenizer) {
    std::fprintf(stderr, "error: %s\n", tokenizer.error().c_str());
    return 1;
  }
  auto tpl = llmi::chat::ChatTemplate::load(args.model_dir + "/tokenizer_config.json");
  if (!tpl) {
    std::fprintf(stderr, "error: %s (this model may not have a chat template)\n", tpl.error().c_str());
    return 1;
  }

  const auto& cfg = transformer->config();
  auto cache = transformer->new_cache(cfg.max_position_embeddings);
  if (!cache) {
    std::fprintf(stderr, "error: %s\n", cache.error().c_str());
    return 1;
  }
  std::vector<llmi::TokenId> cached_tokens;  // exactly what forward_cached() has already processed
  std::mt19937_64 rng(args.seed);
  llmi::SamplingConfig sampling;
  sampling.temperature = args.temperature;
  sampling.top_k = args.top_k;
  sampling.top_p = args.top_p;

  std::vector<llmi::chat::Message> messages;
  if (!args.system.empty()) messages.push_back({"system", args.system});

  std::fprintf(stderr, "llmi-chat: %s loaded. Type a message and press enter (Ctrl-D to quit).\n",
               args.model_dir.c_str());
  std::string line;
  while (std::fprintf(stderr, "> "), std::getline(std::cin, line)) {
    if (line.empty()) continue;
    messages.push_back({"user", line});

    auto prompt_text = tpl->render(messages, /*add_generation_prompt=*/true);
    if (!prompt_text) {
      std::fprintf(stderr, "error: %s\n", prompt_text.error().c_str());
      return 1;
    }
    std::size_t dropped = 0;
    auto full_ids = tokenizer->encode(*prompt_text, /*parse_special=*/true, &dropped);
    if (!full_ids) {
      std::fprintf(stderr, "error: %s\n", full_ids.error().c_str());
      return 1;
    }

    const std::size_t reuse = common_prefix_length(cached_tokens, full_ids.value());
    if (reuse < cached_tokens.size()) {
      // The cached prefix no longer matches (most likely a BPE merge that
      // spans the old/new text boundary): start this turn's prefill over.
      cache->reset();
      cached_tokens.clear();
    }
    const std::vector<llmi::TokenId> new_tokens(full_ids->begin() + static_cast<long>(cached_tokens.size()),
                                                full_ids->end());
    cached_tokens = full_ids.value();

    auto logits = transformer->forward_cached(new_tokens, *cache);
    if (!logits) {
      std::fprintf(stderr, "error: %s\n", logits.error().c_str());
      return 1;
    }

    std::vector<llmi::TokenId> reply;
    llmi::TokenId next = llmi::sample(logits->data(), logits->size(), sampling, rng);
    for (std::size_t step = 0; step < args.max_new; ++step) {
      bool stop = false;
      for (const auto eos : cfg.eos_token_ids) stop = stop || next == eos;
      if (stop) break;
      reply.push_back(next);
      cached_tokens.push_back(next);
      auto text = tokenizer->decode({next});
      if (text) {
        std::fwrite(text->data(), 1, text->size(), stdout);
        std::fflush(stdout);
      }
      if (step + 1 == args.max_new) break;
      auto step_logits = transformer->forward_cached({next}, *cache);
      if (!step_logits) {
        std::fprintf(stderr, "\nerror: %s\n", step_logits.error().c_str());
        return 1;
      }
      next = llmi::sample(step_logits->data(), step_logits->size(), sampling, rng);
    }
    std::printf("\n");
    auto reply_text = tokenizer->decode(reply);
    messages.push_back({"assistant", reply_text ? *reply_text : ""});
  }
  return 0;
}
