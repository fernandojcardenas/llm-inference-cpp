#include "llmi/model/kernels.hpp"

#include <algorithm>
#include <cmath>

namespace llmi::kernels {

float dot(const float* a, const float* b, std::size_t n) {
  float acc[8] = {};
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    for (std::size_t k = 0; k < 8; ++k) acc[k] += a[i + k] * b[i + k];
  }
  float tail = 0;
  for (; i < n; ++i) tail += a[i] * b[i];
  return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail;
}

void matmul(const float* x, std::size_t rows, std::size_t in, const float* w, std::size_t out, const float* bias,
            float* y) {
  // Weight row outermost: each row of w is read once and reused for every
  // input row while it is in cache.
  for (std::size_t o = 0; o < out; ++o) {
    const float* wr = w + o * in;
    const float b = bias != nullptr ? bias[o] : 0.0F;
    for (std::size_t r = 0; r < rows; ++r) y[r * out + o] = dot(x + r * in, wr, in) + b;
  }
}

void rmsnorm(const float* x, const float* weight, std::size_t n, float eps, float* y) {
  double ss = 0;
  for (std::size_t i = 0; i < n; ++i) ss += static_cast<double>(x[i]) * x[i];
  const auto scale = static_cast<float>(1.0 / std::sqrt(ss / static_cast<double>(n) + eps));
  for (std::size_t i = 0; i < n; ++i) y[i] = weight[i] * (x[i] * scale);
}

void rope(float* v, std::size_t head_dim, std::size_t pos, const float* inv_freq) {
  const std::size_t half = head_dim / 2;
  for (std::size_t i = 0; i < half; ++i) {
    // As the reference computes it: the angle in float32, then cos and sin.
    const float angle = static_cast<float>(pos) * inv_freq[i];
    const float c = std::cos(angle);
    const float s = std::sin(angle);
    const float a = v[i];
    const float b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = b * c + a * s;
  }
}

void softmax(float* x, std::size_t n) {
  if (n == 0) return;
  const float mx = *std::max_element(x, x + n);
  double sum = 0;
  for (std::size_t i = 0; i < n; ++i) {
    x[i] = std::exp(x[i] - mx);
    sum += x[i];
  }
  const auto inv = static_cast<float>(1.0 / sum);
  for (std::size_t i = 0; i < n; ++i) x[i] *= inv;
}

float silu(float x) { return x / (1.0F + std::exp(-x)); }

}  // namespace llmi::kernels
