#pragma once

#include <cstdint>
#include <vector>

#include "llmi/model/model.hpp"
#include "llmi/model/quant.hpp"
#include "llmi/model/weight.hpp"
#include "llmi/tokenizer/tokenizer.hpp"
#include "llmi/util/result.hpp"

namespace llmi {

// Intermediate values of one forward pass, for checking against a reference:
// states[0] is the token embeddings, states[1..L] the output of each layer,
// states[L+1] the final RMSNorm. Each is tokens x hidden_size, row-major.
struct ForwardTrace {
  std::vector<std::vector<float>> states;
};

// Per-layer key/value history for incremental decoding. Position `length`
// (0-based) is the next slot forward_cached() will fill; everything before
// it was computed by an earlier call and is reused, not recomputed.
//
// A cache only ever grows by appending, so its contents at any point are
// exactly the K/V that Transformer::forward() would have computed for the
// same token sequence from scratch: forward_cached() must reproduce
// forward()'s baseline token for token (ADR 0004).
class KVCache {
 public:
  [[nodiscard]] std::size_t length() const { return length_; }
  [[nodiscard]] std::size_t capacity() const { return capacity_; }
  void reset() { length_ = 0; }

 private:
  friend class Transformer;
  std::size_t capacity_ = 0;
  std::size_t length_ = 0;
  // k_[layer] / v_[layer]: capacity_ * num_kv_heads * head_dim, row-major by position.
  std::vector<std::vector<float>> k_, v_;
};

// A decoder-only transformer (Llama / Qwen2) in float32.
//
//   x = embed[token]
//   per layer:  x += Wo * attention(rope(Wq*rmsnorm(x)), rope(Wk*...), Wv*...)   (causal, grouped-query)
//               x += Wdown * (silu(Wgate*rmsnorm(x)) * (Wup*rmsnorm(x)))
//   logits = lm_head * rmsnorm(x)
//
// forward() recomputes the whole sequence every call (M2's baseline, kept for
// cross-checking and for --dump). forward_cached() keeps past keys/values in
// a KVCache and only computes the new tokens (M3): the KV cache test suite
// (transformer_test.cpp) checks it reproduces forward() exactly.
class Transformer {
 public:
  // Converts every weight to float32 (BF16 and F16 convert exactly), then,
  // when weight_type isn't Type::F32, quantizes every weight matrix the
  // forward pass matmuls against (M5: Q8_0 or Q4_0 -- see
  // docs/quantization.md and ADR 0006). RMSNorm weights, biases and
  // inv_freq stay float32 regardless: they're small and numerically
  // sensitive, the same choice llama.cpp makes for Q8_0/Q4_0.
  static Result<Transformer> load(const Model& model, quant::Type weight_type = quant::Type::F32);

  [[nodiscard]] const ModelConfig& config() const { return cfg_; }

  // Total bytes of quantizable weight storage (matmul matrices + the
  // embedding/lm_head table), at whichever type load() was given -- used to
  // report M5's memory table.
  [[nodiscard]] std::size_t weight_bytes() const;

  // Logits for the last position (vocab_size values), or for every position
  // (tokens x vocab_size) when all_positions is set.
  [[nodiscard]] Result<std::vector<float>> forward(const std::vector<TokenId>& tokens, bool all_positions = false,
                                                   ForwardTrace* trace = nullptr) const;

  // Greedy decoding: append the highest-scoring token max_new times (lowest id
  // on ties), stopping early at a stop token unless stop_ids is empty.
  [[nodiscard]] Result<std::vector<TokenId>> generate_greedy(std::vector<TokenId> tokens, std::size_t max_new,
                                                             const std::vector<TokenId>& stop_ids) const;

  // A cache sized to hold up to max_len positions (capped at the model's
  // context length).
  [[nodiscard]] Result<KVCache> new_cache(std::size_t max_len) const;

  // Processes only `tokens` (new positions, continuing where the cache left
  // off), appends their keys/values to cache, and returns logits for the
  // last new position (or every new position when all_positions is set).
  // tokens.size() can be 1 (one decode step) or more (prefilling a prompt).
  [[nodiscard]] Result<std::vector<float>> forward_cached(const std::vector<TokenId>& tokens, KVCache& cache,
                                                          bool all_positions = false,
                                                          ForwardTrace* trace = nullptr) const;

  // Greedy decoding using a KV cache: the prompt is prefilled once, then each
  // step computes only the one new token. Produces the same tokens as
  // generate_greedy on the same input.
  [[nodiscard]] Result<std::vector<TokenId>> generate_greedy_cached(const std::vector<TokenId>& prompt,
                                                                    std::size_t max_new,
                                                                    const std::vector<TokenId>& stop_ids) const;

 private:
  struct Layer {
    std::vector<float> attn_norm, bq, bk, bv;
    Weight wq, wk, wv, wo;
    std::vector<float> mlp_norm;
    Weight w_gate, w_up, w_down;
  };
  ModelConfig cfg_;
  Weight embed_;
  std::vector<float> final_norm_;
  Weight lm_head_;  // empty when tied to embed_
  std::vector<Layer> layers_;
  std::vector<float> inv_freq_;
};

// Index of the largest value; the first one on ties (as torch.argmax).
std::size_t argmax(const float* x, std::size_t n);

}  // namespace llmi
