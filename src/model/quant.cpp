#include "llmi/model/quant.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "llmi/util/thread_pool.hpp"

namespace llmi::quant {

namespace {

// Same threshold and reasoning as kernels::matmul (docs/performance.md):
// below this many multiply-adds, dispatching to the thread pool costs more
// than it saves.
constexpr std::size_t kMinWorkForThreading = 200'000;

std::size_t bytes_per_row(std::size_t in, Type type) {
  return type == Type::Q4_0 ? (in + 1) / 2 : in;
}

}  // namespace

std::string_view type_name(Type t) {
  switch (t) {
    case Type::F32:
      return "f32";
    case Type::Q8_0:
      return "q8_0";
    case Type::Q4_0:
      return "q4_0";
  }
  return "unknown";
}

QuantizedMatrix quantize(const float* w, std::size_t out, std::size_t in, Type type) {
  QuantizedMatrix m;
  m.type = type;
  m.out = out;
  m.in = in;
  m.blocks_per_row = (in + kBlockSize - 1) / kBlockSize;
  m.scales.assign(out * m.blocks_per_row, 0.0F);
  m.data.assign(out * bytes_per_row(in, type), 0);

  const int qmax = type == Type::Q4_0 ? 7 : 127;
  const int qmin = type == Type::Q4_0 ? -8 : -127;

  for (std::size_t r = 0; r < out; ++r) {
    const float* row = w + r * in;
    std::uint8_t* qrow = m.data.data() + r * bytes_per_row(in, type);
    for (std::size_t b = 0; b < m.blocks_per_row; ++b) {
      const std::size_t start = b * kBlockSize;
      const std::size_t len = std::min(kBlockSize, in - start);
      float amax = 0.0F;
      float signed_extreme = 0.0F;  // the value (not just magnitude) at the block's largest |value|
      for (std::size_t i = 0; i < len; ++i) {
        const float v = row[start + i];
        if (std::abs(v) > amax) {
          amax = std::abs(v);
          signed_extreme = v;
        }
      }
      float scale;
      if (type == Type::Q4_0) {
        // Scale's sign matches the block's extreme value, same convention
        // GGUF's own Q4_0 uses (ggml's quantize_row_q4_0_reference: d =
        // max/-8, where `max` keeps its sign). Q4_0's range is asymmetric
        // (-8..7: one more negative code than positive, since 0 has to be
        // representable), so without this, whichever sign happens to hold
        // the block's single most important value -- the one with the
        // largest rounding error if clipped -- would always land on the
        // *missing* code (q=+8, clamped down to 7) and lose up to a whole
        // quantization step. Letting the scale's sign track the extreme
        // value puts that value at q=-8 instead, which always exists, so
        // it reconstructs exactly regardless of its sign (verified by
        // Quant.Q4_0ExtremeValueInEachBlockRoundTripsExactly).
        scale = signed_extreme / -8.0F;
      } else {
        scale = amax / 127.0F;  // Q8_0's range (-127..127) is already symmetric: no such case to handle
      }
      if (scale == 0.0F) scale = 1.0F;  // an all-zero block; every q below will be 0 regardless
      m.scales[r * m.blocks_per_row + b] = scale;
      for (std::size_t i = 0; i < len; ++i) {
        const std::size_t idx = start + i;
        const long rounded = std::lround(static_cast<double>(row[idx]) / scale);
        const int q = std::clamp(static_cast<int>(rounded), qmin, qmax);
        if (type == Type::Q8_0) {
          qrow[idx] = static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
        } else {
          const auto nibble = static_cast<std::uint8_t>(q + 8);
          std::uint8_t& byte = qrow[idx / 2];
          if (idx % 2 == 0) {
            byte = static_cast<std::uint8_t>((byte & 0xF0U) | nibble);
          } else {
            byte = static_cast<std::uint8_t>((byte & 0x0FU) | static_cast<std::uint8_t>(nibble << 4U));
          }
        }
      }
    }
  }
  return m;
}

void dequantize_row(const QuantizedMatrix& m, std::size_t row, float* dst) {
  const std::uint8_t* qrow = m.data.data() + row * bytes_per_row(m.in, m.type);
  const float* scales = m.scales.data() + row * m.blocks_per_row;
  for (std::size_t idx = 0; idx < m.in; ++idx) {
    const float scale = scales[idx / kBlockSize];
    if (m.type == Type::Q8_0) {
      dst[idx] = static_cast<float>(static_cast<std::int8_t>(qrow[idx])) * scale;
    } else {
      const std::uint8_t byte = qrow[idx / 2];
      const auto nibble = static_cast<int>((idx % 2 == 0) ? (byte & 0x0FU) : (byte >> 4U));
      dst[idx] = static_cast<float>(nibble - 8) * scale;
    }
  }
}

void matmul(const float* x, std::size_t rows, std::size_t in, const QuantizedMatrix& w, std::size_t out,
            const float* bias, float* y) {
  const std::size_t row_bytes = bytes_per_row(in, w.type);
  auto run = [&](std::size_t o_begin, std::size_t o_end) {
    for (std::size_t o = o_begin; o < o_end; ++o) {
      const std::uint8_t* qrow = w.data.data() + o * row_bytes;
      const float* scales = w.scales.data() + o * w.blocks_per_row;
      const float b = bias != nullptr ? bias[o] : 0.0F;
      for (std::size_t r = 0; r < rows; ++r) {
        const float* xr = x + r * in;
        float acc = 0.0F;
        for (std::size_t blk = 0; blk < w.blocks_per_row; ++blk) {
          const std::size_t start = blk * kBlockSize;
          const std::size_t len = std::min(kBlockSize, in - start);
          float block_sum = 0.0F;
          if (w.type == Type::Q8_0) {
            for (std::size_t i = 0; i < len; ++i) {
              block_sum += xr[start + i] * static_cast<float>(static_cast<std::int8_t>(qrow[start + i]));
            }
          } else {
            for (std::size_t i = 0; i < len; ++i) {
              const std::size_t idx = start + i;
              const std::uint8_t byte = qrow[idx / 2];
              const auto nibble = static_cast<int>((idx % 2 == 0) ? (byte & 0x0FU) : (byte >> 4U));
              block_sum += xr[idx] * static_cast<float>(nibble - 8);
            }
          }
          acc += block_sum * scales[blk];
        }
        y[r * out + o] = acc + b;
      }
    }
  };
  util::ThreadPool::shared().parallel_for(out, kMinWorkForThreading / std::max<std::size_t>(rows * in, 1), run);
}

}  // namespace llmi::quant
