#pragma once

#include <cstddef>
#include <vector>

#include "llmi/model/quant.hpp"

namespace llmi {

// A weight matrix (out x in) that is either plain float32 or block-quantized
// (M5). Transformer's layers hold these instead of bare std::vector<float>
// so forward()/forward_cached() can call matmul()/row() without knowing
// which storage backs a given weight. Selecting quant::Type::F32 keeps the
// original std::vector<float> storage and calls kernels::matmul directly, so
// it reproduces M1-M4's bit-exact behavior unchanged; only requesting Q8_0
// or Q4_0 at load time switches a weight to quant::matmul's inline-dequant
// path (ADR 0006).
class Weight {
 public:
  Weight() = default;
  static Weight from_float(std::vector<float> w);
  static Weight quantized(const std::vector<float>& w, std::size_t out, std::size_t in, quant::Type type);

  [[nodiscard]] bool empty() const;
  [[nodiscard]] std::size_t byte_size() const;
  [[nodiscard]] quant::Type type() const { return type_; }

  // y[r][o] = sum_i x[r][i] * this[o][i] (+ bias[o]), for every row r.
  void matmul(const float* x, std::size_t rows, std::size_t in, std::size_t out, const float* bias, float* y) const;

  // Copies row `index` (length `in`) into dst -- used for the embedding
  // table lookup, which is a gather rather than a matmul.
  void row(std::size_t index, std::size_t in, float* dst) const;

 private:
  quant::Type type_ = quant::Type::F32;
  std::vector<float> f32_;
  quant::QuantizedMatrix q_;
};

}  // namespace llmi
