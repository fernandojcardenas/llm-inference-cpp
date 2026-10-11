#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "llmi/grammar/grammar.hpp"
#include "llmi/grammar/matcher.hpp"
#include "llmi/model/sampling.hpp"
#include "llmi/tokenizer/tokenizer.hpp"
#include "llmi/util/result.hpp"

namespace llmi::grammar {

// Constrained decoding, part 4: choosing tokens (M8, ADR 0009).
//
// Vocabulary: every token's raw bytes and the prefix tree over them, built
// once per tokenizer. Special tokens get no bytes, so a grammar never allows
// them. End-of-sequence ids are handled separately: allowed exactly when the
// grammar is complete, and never as text, even in a vocabulary where the
// same id also decodes to bytes.
struct Vocabulary {
  explicit Vocabulary(const Tokenizer& tokenizer);
  std::vector<std::string> token_bytes;
  TokenTrie trie;
};

// One constrained generation. pick() chooses the next token from the
// model's logits; accept() advances the grammar past it.
//
// Disallowed tokens get a logit of -infinity BEFORE temperature, top-k and
// top-p are applied, so sampling is over the allowed tokens only and its
// probabilities are the model's own, renormalized over that set.
//
// Greedy decoding (temperature <= 0) first checks the unconstrained argmax:
// if the grammar allows it, it is also the argmax of the masked logits, and
// the full mask is skipped. Sampling always computes the full mask, because
// top-k and top-p cut-offs depend on which tokens survive it.
class Constraint {
 public:
  Constraint(std::shared_ptr<const Grammar> grammar, std::shared_ptr<const Vocabulary> vocab,
             std::vector<TokenId> eos_ids);

  // Modifies `logits` (masks disallowed tokens). Fails if no token can
  // continue the grammar -- possible if the grammar needs a character no
  // token in this vocabulary contains.
  Result<TokenId> pick(std::vector<float>& logits, const SamplingConfig& cfg, std::mt19937_64& rng);

  // Advances past `id`. False if the grammar does not allow it (pick() never
  // returns such a token).
  bool accept(TokenId id);

  // The output is a complete sentence and the grammar allows nothing more:
  // generation can stop without asking the model.
  [[nodiscard]] bool finished() const;
  [[nodiscard]] bool complete() const { return matcher_.complete(); }

  struct Stats {
    std::size_t picks = 0;
    std::size_t full_masks = 0;        // picks that needed the whole vocabulary checked
    std::size_t trie_nodes_visited = 0;
    double mask_seconds = 0.0;         // time spent deciding which tokens are allowed
  };
  [[nodiscard]] const Stats& stats() const { return stats_; }

 private:
  [[nodiscard]] bool allowed(TokenId id) const;
  [[nodiscard]] bool is_eos(TokenId id) const;

  std::shared_ptr<const Vocabulary> vocab_;
  Matcher matcher_;
  std::vector<TokenId> eos_ids_;
  std::vector<std::uint8_t> mask_;
  Stats stats_;
};

}  // namespace llmi::grammar
