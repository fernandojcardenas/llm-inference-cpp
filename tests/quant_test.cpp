#include "llmi/model/quant.hpp"

#include <gtest/gtest.h>

#include <random>

#include "llmi/model/kernels.hpp"

using namespace llmi::quant;

namespace {

// Random out x in matrix, zero-centered like real transformer weights (so
// the all-zero-block edge case in quantize() isn't the only thing tested).
std::vector<float> random_matrix(std::size_t out, std::size_t in, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0F, 0.3F);
  std::vector<float> w(out * in);
  for (auto& v : w) v = nd(rng);
  return w;
}

}  // namespace

// -------------------------------------------------------- quantize/dequantize

TEST(Quant, RoundTripQ8_0StaysWithinHalfAStepOfTheOriginal) {
  // Every block's largest |value| maps to +-127 exactly, so the worst-case
  // rounding error for any value in that block is half a quantization step:
  // scale/2 = amax/254. Checked at several `in` sizes, including sizes that
  // aren't a multiple of kBlockSize (32) -- 7 and 33 exercise a single
  // partial block and a full block plus a partial one.
  for (std::size_t in : {1U, 7U, 31U, 32U, 33U, 100U}) {
    constexpr std::size_t out = 3;
    const auto w = random_matrix(out, in, 1);
    const QuantizedMatrix q = quantize(w.data(), out, in, Type::Q8_0);
    ASSERT_EQ(q.blocks_per_row, (in + kBlockSize - 1) / kBlockSize);
    std::vector<float> row(in);
    for (std::size_t r = 0; r < out; ++r) {
      dequantize_row(q, r, row.data());
      for (std::size_t b = 0; b < q.blocks_per_row; ++b) {
        const std::size_t start = b * kBlockSize;
        const std::size_t len = std::min(kBlockSize, in - start);
        float amax = 0.0F;
        for (std::size_t i = 0; i < len; ++i) amax = std::max(amax, std::abs(w[r * in + start + i]));
        const float tol = amax / 254.0F + 1e-6F;
        for (std::size_t i = 0; i < len; ++i) {
          EXPECT_NEAR(row[start + i], w[r * in + start + i], tol) << "in=" << in << " row=" << r << " idx=" << start + i;
        }
      }
    }
  }
}

TEST(Quant, RoundTripQ4_0StaysWithinOneStepOfTheOriginal) {
  // Q4_0's scale takes its sign from the block's extreme value (see
  // quant.hpp), which guarantees THAT value round-trips exactly (checked
  // separately below) -- but a different value that happens to nearly tie
  // it in magnitude with the OPPOSITE sign can still land on the clamp
  // boundary and see a full quantization step (scale = amax/8) of error,
  // not just half a step. Small random blocks (the in=7 case here) make such
  // near-ties more likely to turn up by chance, which is exactly what this
  // bound is for.
  for (std::size_t in : {1U, 7U, 31U, 32U, 33U, 100U}) {
    constexpr std::size_t out = 3;
    const auto w = random_matrix(out, in, 2);
    const QuantizedMatrix q = quantize(w.data(), out, in, Type::Q4_0);
    std::vector<float> row(in);
    for (std::size_t r = 0; r < out; ++r) {
      dequantize_row(q, r, row.data());
      for (std::size_t b = 0; b < q.blocks_per_row; ++b) {
        const std::size_t start = b * kBlockSize;
        const std::size_t len = std::min(kBlockSize, in - start);
        float amax = 0.0F;
        for (std::size_t i = 0; i < len; ++i) amax = std::max(amax, std::abs(w[r * in + start + i]));
        const float tol = amax / 8.0F + 1e-6F;
        for (std::size_t i = 0; i < len; ++i) {
          EXPECT_NEAR(row[start + i], w[r * in + start + i], tol) << "in=" << in << " row=" << r << " idx=" << start + i;
        }
      }
    }
  }
}

TEST(Quant, Q4_0ExtremeValueInEachBlockRoundTripsExactly) {
  // The block's largest-magnitude value (whichever sign it has) always maps
  // to the nibble code -8, which does exist (Q4_0's range is -8..7), so it
  // should reconstruct with zero rounding error -- checked on both a
  // positive-extreme and a negative-extreme block, since getting the scale's
  // sign convention backwards would only break one of the two.
  const float positive_extreme[8] = {0.9F, -0.3F, 0.1F, 0.4F, -0.5F, 0.2F, -0.1F, 0.05F};
  const float negative_extreme[8] = {-0.9F, 0.3F, 0.1F, 0.4F, -0.5F, 0.2F, -0.1F, 0.05F};
  for (const float* w : {positive_extreme, negative_extreme}) {
    const QuantizedMatrix q = quantize(w, 1, 8, Type::Q4_0);
    std::vector<float> row(8);
    dequantize_row(q, 0, row.data());
    EXPECT_FLOAT_EQ(row[0], w[0]) << "extreme value " << w[0];
  }
}

TEST(Quant, AllZeroBlockRoundTripsToExactlyZero) {
  // A block of all zeros would otherwise divide by zero computing its scale;
  // quantize() must special-case it rather than propagate a NaN/Inf scale.
  constexpr std::size_t in = 40;
  std::vector<float> w(in, 0.0F);
  for (Type t : {Type::Q8_0, Type::Q4_0}) {
    const QuantizedMatrix q = quantize(w.data(), 1, in, t);
    std::vector<float> row(in);
    dequantize_row(q, 0, row.data());
    for (float v : row) EXPECT_EQ(v, 0.0F);
  }
}

TEST(Quant, ByteSizeMatchesTheDocumentedCompressionRatio) {
  // Q8_0: 1 byte/weight + one f32 scale per 32 weights -> ~1.125 bytes/weight
  // (~3.55x smaller than f32's 4 bytes). Q4_0: 0.5 bytes/weight + the same
  // scale overhead -> ~0.625 bytes/weight (~6.4x smaller). Checked on a shape
  // large enough that the per-row scale overhead is close to its asymptotic
  // fraction (docs/quantization.md quotes these ratios).
  constexpr std::size_t out = 100, in = 3200;  // in is a multiple of kBlockSize, no partial-block rounding
  const auto w = random_matrix(out, in, 3);
  const auto f32_bytes = static_cast<double>(out * in * sizeof(float));
  const QuantizedMatrix q8 = quantize(w.data(), out, in, Type::Q8_0);
  const QuantizedMatrix q4 = quantize(w.data(), out, in, Type::Q4_0);
  EXPECT_NEAR(f32_bytes / static_cast<double>(q8.byte_size()), 3.55, 0.05);
  EXPECT_NEAR(f32_bytes / static_cast<double>(q4.byte_size()), 6.4, 0.1);
}

// --------------------------------------------------------------------- matmul

TEST(Quant, MatmulMatchesDequantizeThenDotProduct) {
  // quant::matmul's inline block-wise accumulation (scale applied once per
  // block rather than once per element) is mathematically the same sum as
  // dequantizing the whole row first and taking a float32 dot product, just
  // reordered -- so it should match that reference closely, not necessarily
  // bit-for-bit (unlike kernels::matmul vs kernels::dot, which are exact).
  constexpr std::size_t rows = 3, in = 300, out = 50;
  std::mt19937 rng(4);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> x(rows * in), bias(out);
  for (auto& v : x) v = u(rng);
  for (auto& v : bias) v = u(rng);
  const auto w = random_matrix(out, in, 5);

  for (Type t : {Type::Q8_0, Type::Q4_0}) {
    const QuantizedMatrix q = quantize(w.data(), out, in, t);
    std::vector<float> y(rows * out);
    matmul(x.data(), rows, in, q, out, bias.data(), y.data());

    std::vector<float> row(in);
    for (std::size_t o = 0; o < out; ++o) {
      dequantize_row(q, o, row.data());
      for (std::size_t r = 0; r < rows; ++r) {
        const float want = llmi::kernels::dot(x.data() + r * in, row.data(), in) + bias[o];
        EXPECT_NEAR(y[r * out + o], want, 1e-3F) << (t == Type::Q8_0 ? "q8_0" : "q4_0") << " r=" << r << " o=" << o;
      }
    }
  }
}

TEST(Quant, MatmulAboveTheThreadingThresholdMatchesARowByRowReference) {
  // Large enough (rows * in * out) that the thread pool actually splits the
  // output range -- the same reasoning as kernels::matmul's equivalent test.
  constexpr std::size_t rows = 3, in = 300, out = 4000;
  std::mt19937 rng(6);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> x(rows * in);
  for (auto& v : x) v = u(rng);
  const auto w = random_matrix(out, in, 7);
  const QuantizedMatrix q = quantize(w.data(), out, in, Type::Q8_0);

  std::vector<float> y(rows * out);
  matmul(x.data(), rows, in, q, out, nullptr, y.data());

  std::vector<float> row(in);
  for (std::size_t o = 0; o < out; ++o) {
    dequantize_row(q, o, row.data());
    for (std::size_t r = 0; r < rows; ++r) {
      const float want = llmi::kernels::dot(x.data() + r * in, row.data(), in);
      ASSERT_NEAR(y[r * out + o], want, 1e-3F) << "r=" << r << " o=" << o;
    }
  }
}
