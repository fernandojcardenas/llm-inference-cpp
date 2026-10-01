#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace llmi::quant {

// Block size shared by both quantized formats below, matching GGUF's own
// Q8_0/Q4_0 block size -- so a weight this engine quantizes and one
// llama.cpp quantizes from the same float32 source are directly comparable
// (same granularity, same symmetric-scale convention), which is what M5's
// perplexity and speed comparisons rest on.
inline constexpr std::size_t kBlockSize = 32;

enum class Type : std::uint8_t { F32, Q8_0, Q4_0 };

std::string_view type_name(Type t);

// A quantized out x in weight matrix (the same out x in layout
// kernels::matmul's `w` uses). Each row is split into
// ceil(in / kBlockSize) blocks; each block has its own float32 scale and
// `kBlockSize` (or fewer, for a final partial block) quantized values.
// Both formats are symmetric, scale-only (no zero-point): scale is derived
// from the block's largest magnitude, so a weight of exactly 0 always
// reconstructs to exactly 0 -- important since transformer weights are
// heavily zero-centered.
//   Q8_0: one signed int8 per weight (range -127..127), value = q * scale.
//   Q4_0: one unsigned nibble per weight, stored with a +8 bias (0..15) so
//         two weights pack into a byte; value = (q - 8) * scale, q in 0..15
//         so the reconstructed range is -8..7. This range is asymmetric (one
//         more negative code than positive, since 0 must be representable),
//         so Q4_0's scale takes its SIGN from the block's largest-magnitude
//         value (matching GGUF's own Q4_0 convention): that value always
//         lands on the code that does exist (-8) and round-trips exactly,
//         regardless of its sign, rather than ever needing the missing +8.
struct QuantizedMatrix {
  Type type = Type::F32;
  std::size_t out = 0;
  std::size_t in = 0;
  std::size_t blocks_per_row = 0;
  std::vector<float> scales;       // out * blocks_per_row
  std::vector<std::uint8_t> data;  // Q8_0: out*in bytes; Q4_0: out*ceil(in/2) bytes

  [[nodiscard]] bool empty() const { return out == 0 || in == 0; }
  [[nodiscard]] std::size_t byte_size() const { return data.size() + scales.size() * sizeof(float); }
};

// Quantizes an out x in row-major float32 matrix. type must be Q8_0 or Q4_0.
QuantizedMatrix quantize(const float* w, std::size_t out, std::size_t in, Type type);

// Reconstructs one row (length m.in) into dst.
void dequantize_row(const QuantizedMatrix& m, std::size_t row, float* dst);

// y[r][o] = sum_i x[r][i] * dequantized(w[o][i]) (+ bias[o]), for every row
// r in [0, rows). Dequantizes each block inline (never materializes a full
// float row) and threads over the output dimension exactly like
// kernels::matmul (M4): different o's write disjoint elements of y, so this
// needs no locking. Not bit-exact against a float32 matmul of the
// dequantized weights -- only kernels::matmul (Type::F32, no quantization)
// carries that guarantee -- but checked against it within quantization's own
// error bound (tests/quant_test.cpp).
void matmul(const float* x, std::size_t rows, std::size_t in, const QuantizedMatrix& w, std::size_t out,
            const float* bias, float* y);

}  // namespace llmi::quant
