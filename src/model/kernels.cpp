#include "llmi/model/kernels.hpp"

#include <algorithm>
#include <cmath>

#include "llmi/util/thread_pool.hpp"

// Platform SIMD for dot(), the innermost loop of matmul (which is nearly all
// of the forward pass's time -- see docs/performance.md). Picked at compile
// time, not runtime-dispatched: there is no AVX2-capable-but-not-used
// fallback path to maintain, since the project already documents its
// supported platforms (an x86_64 Linux/macOS dev machine with AVX2, or
// Apple Silicon, where NEON is simply part of the base ISA). A CPU without
// AVX2 falls back to the portable scalar version below, just slower.
#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)
#define LLMI_HAVE_AVX2 1
#include <immintrin.h>
#elif defined(__aarch64__) && defined(__ARM_NEON)
#define LLMI_HAVE_NEON 1
#include <arm_neon.h>
#endif

namespace llmi::kernels {

namespace {
// Below this many multiply-adds, matmul runs on the calling thread alone:
// dispatching to the thread pool costs more in synchronization (a handful of
// mutex/condvar round trips) than it could possibly save. Chosen from
// tools/bench_kernels.cpp: a 1x576x576 matmul (~330K MACs, ~0.04ms serial on
// this machine) is right at the edge where threading starts paying for
// itself on a 2-core machine; see docs/performance.md.
constexpr std::size_t kMinWorkForThreading = 200'000;
}  // namespace

namespace {

// The original portable version: 8 independent scalar accumulators so the
// compiler can autovectorize without -ffast-math. Used directly when no
// SIMD path is compiled in, and as the tail handler (fewer than one SIMD
// width of elements left) for both SIMD paths below.
float dot_scalar(const float* a, const float* b, std::size_t n) {
  float acc[8] = {};
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    for (std::size_t k = 0; k < 8; ++k) acc[k] += a[i + k] * b[i + k];
  }
  float tail = 0;
  for (; i < n; ++i) tail += a[i] * b[i];
  return ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7])) + tail;
}

}  // namespace

#if defined(LLMI_HAVE_AVX2)

float dot(const float* a, const float* b, std::size_t n) {
  // Two 8-wide accumulators (16 lanes total) so independent FMA chains can
  // overlap, the same idea as dot_scalar's 8 scalar accumulators.
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 16 <= n; i += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
  }
  const __m256 acc = _mm256_add_ps(acc0, acc1);
  // Horizontal sum of the 8 lanes.
  const __m128 lo = _mm256_castps256_ps128(acc);
  const __m128 hi = _mm256_extractf128_ps(acc, 1);
  __m128 sum4 = _mm_add_ps(lo, hi);
  sum4 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
  sum4 = _mm_add_ss(sum4, _mm_shuffle_ps(sum4, sum4, 1));
  return _mm_cvtss_f32(sum4) + dot_scalar(a + i, b + i, n - i);
}

#elif defined(LLMI_HAVE_NEON)

float dot(const float* a, const float* b, std::size_t n) {
  // Two 4-wide accumulators (8 lanes total), mirroring the AVX2 path above.
  float32x4_t acc0 = vdupq_n_f32(0.0F);
  float32x4_t acc1 = vdupq_n_f32(0.0F);
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
    acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
  }
  const float32x4_t acc = vaddq_f32(acc0, acc1);
  return vaddvq_f32(acc) + dot_scalar(a + i, b + i, n - i);
}

#else

float dot(const float* a, const float* b, std::size_t n) { return dot_scalar(a, b, n); }

#endif

void matmul(const float* x, std::size_t rows, std::size_t in, const float* w, std::size_t out, const float* bias,
            float* y) {
  // Weight row outermost: each row of w is read once and reused for every
  // input row while it is in cache. Different o's write disjoint elements of
  // y (interleaved by `out`, but never the same one), so splitting the o
  // range across threads needs no locking and changes no floating-point
  // operation's order -- the result is bit-for-bit identical to the serial
  // loop, just computed by more than one thread at once.
  auto run = [&](std::size_t o_begin, std::size_t o_end) {
    for (std::size_t o = o_begin; o < o_end; ++o) {
      const float* wr = w + o * in;
      const float b = bias != nullptr ? bias[o] : 0.0F;
      for (std::size_t r = 0; r < rows; ++r) y[r * out + o] = dot(x + r * in, wr, in) + b;
    }
  };
  util::ThreadPool::shared().parallel_for(out, kMinWorkForThreading / std::max<std::size_t>(rows * in, 1), run);
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
