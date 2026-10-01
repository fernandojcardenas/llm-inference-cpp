#pragma once

#include <cstddef>
#include <random>
#include <vector>

#include "llmi/tokenizer/tokenizer.hpp"

namespace llmi {

// Temperature, top-k and top-p (nucleus) sampling, applied in that order
// (Hugging Face's default warper order: TemperatureLogitsWarper, then
// TopKLogitsWarper, then TopPLogitsWarper).
//
// temperature <= 0 is this engine's own convention for "sample greedily":
// it skips the softmax and returns the argmax, matching generate_greedy.
// Hugging Face's `do_sample=True` does not accept temperature 0 at all,
// so there is nothing to cross-check there; it is documented as our choice
// (ADR 0004), not a reproduction of reference behaviour.
struct SamplingConfig {
  float temperature = 1.0F;
  std::size_t top_k = 0;  // 0 = no top-k filter
  float top_p = 1.0F;     // 1 = no nucleus filter
};

// The probability engine sample() would draw from: softmax(logits /
// temperature), then zeroed outside the top-k set, then zeroed outside the
// smallest nucleus whose cumulative probability reaches top_p (always
// keeping at least the single most likely token), renormalised to sum to 1.
// Exposed separately so the filtering itself can be checked without a
// random draw involved.
std::vector<float> sampling_probabilities(const float* logits, std::size_t n, const SamplingConfig& cfg);

// Draws one token id from sampling_probabilities(logits, n, cfg) using rng.
// rng is the engine's own generator (std::mt19937_64): a seed reproduces a
// run of this engine exactly, but this is not a bit-for-bit match to any
// particular reference implementation's RNG stream, which uses a different
// algorithm (see docs/sampling.md).
TokenId sample(const float* logits, std::size_t n, const SamplingConfig& cfg, std::mt19937_64& rng);

}  // namespace llmi
