#pragma once

#include <cstdint>
#include <vector>

#include "llmi/model/model.hpp"
#include "llmi/tokenizer/tokenizer.hpp"
#include "llmi/util/result.hpp"

namespace llmi {

// Intermediate values of one forward pass, for checking against a reference:
// states[0] is the token embeddings, states[1..L] the output of each layer,
// states[L+1] the final RMSNorm. Each is tokens x hidden_size, row-major.
struct ForwardTrace {
  std::vector<std::vector<float>> states;
};

// A decoder-only transformer (Llama / Qwen2) in float32.
//
//   x = embed[token]
//   per layer:  x += Wo * attention(rope(Wq*rmsnorm(x)), rope(Wk*...), Wv*...)   (causal, grouped-query)
//               x += Wdown * (silu(Wgate*rmsnorm(x)) * (Wup*rmsnorm(x)))
//   logits = lm_head * rmsnorm(x)
//
// M2 recomputes the whole sequence on every call; the key/value cache is M3.
class Transformer {
 public:
  // Converts every weight to float32 (BF16 and F16 convert exactly).
  static Result<Transformer> load(const Model& model);

  [[nodiscard]] const ModelConfig& config() const { return cfg_; }

  // Logits for the last position (vocab_size values), or for every position
  // (tokens x vocab_size) when all_positions is set.
  [[nodiscard]] Result<std::vector<float>> forward(const std::vector<TokenId>& tokens, bool all_positions = false,
                                                   ForwardTrace* trace = nullptr) const;

  // Greedy decoding: append the highest-scoring token max_new times (lowest id
  // on ties), stopping early at a stop token unless stop_ids is empty.
  [[nodiscard]] Result<std::vector<TokenId>> generate_greedy(std::vector<TokenId> tokens, std::size_t max_new,
                                                             const std::vector<TokenId>& stop_ids) const;

 private:
  struct Layer {
    std::vector<float> attn_norm, wq, wk, wv, bq, bk, bv, wo;
    std::vector<float> mlp_norm, w_gate, w_up, w_down;
  };
  ModelConfig cfg_;
  std::vector<float> embed_;
  std::vector<float> final_norm_;
  std::vector<float> lm_head_;  // empty when tied to embed_
  std::vector<Layer> layers_;
  std::vector<float> inv_freq_;
};

// Index of the largest value; the first one on ties (as torch.argmax).
std::size_t argmax(const float* x, std::size_t n);

}  // namespace llmi
