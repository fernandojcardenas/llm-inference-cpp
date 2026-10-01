#include "llmi/model/sampling.hpp"

#include <algorithm>
#include <numeric>

#include "llmi/model/kernels.hpp"
#include "llmi/model/transformer.hpp"

namespace llmi {

std::vector<float> sampling_probabilities(const float* logits, std::size_t n, const SamplingConfig& cfg) {
  std::vector<float> p(logits, logits + n);
  if (cfg.temperature <= 0.0F) {
    // This engine's convention: temperature <= 0 means "sample greedily".
    // There is no softmax to speak of; put all mass on the argmax.
    std::fill(p.begin(), p.end(), 0.0F);
    p[argmax(logits, n)] = 1.0F;
    return p;
  }
  for (float& x : p) x /= cfg.temperature;
  kernels::softmax(p.data(), n);

  // Indices sorted by probability, descending.
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0U);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return p[a] > p[b]; });

  // Top-k: keep only the k highest-probability entries.
  const std::size_t k = (cfg.top_k == 0 || cfg.top_k > n) ? n : cfg.top_k;

  // Top-p (nucleus): among those k, keep the smallest prefix (in descending
  // order) whose cumulative probability reaches top_p. Always keep at least
  // the single most likely token, even if its own probability exceeds top_p.
  std::size_t keep = k;
  if (cfg.top_p < 1.0F) {
    double cumulative = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      cumulative += p[order[i]];
      if (cumulative >= static_cast<double>(cfg.top_p)) {
        keep = i + 1;
        break;
      }
    }
    keep = std::min(keep, k);
    if (keep == 0) keep = 1;
  }

  std::vector<float> filtered(n, 0.0F);
  double sum = 0.0;
  for (std::size_t i = 0; i < keep; ++i) {
    filtered[order[i]] = p[order[i]];
    sum += p[order[i]];
  }
  if (sum > 0.0) {
    for (float& x : filtered) x = static_cast<float>(static_cast<double>(x) / sum);
  }
  return filtered;
}

TokenId sample(const float* logits, std::size_t n, const SamplingConfig& cfg, std::mt19937_64& rng) {
  if (cfg.temperature <= 0.0F) return static_cast<TokenId>(argmax(logits, n));
  const std::vector<float> p = sampling_probabilities(logits, n, cfg);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const double target = u(rng);
  double cumulative = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    cumulative += p[i];
    if (cumulative >= target) return static_cast<TokenId>(i);
  }
  // Floating-point rounding may leave cumulative just under 1: fall back to
  // the last nonzero entry rather than an out-of-range id.
  for (std::size_t i = n; i-- > 0;) {
    if (p[i] > 0.0F) return static_cast<TokenId>(i);
  }
  return static_cast<TokenId>(argmax(logits, n));
}

}  // namespace llmi
