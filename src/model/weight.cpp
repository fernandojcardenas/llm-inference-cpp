#include "llmi/model/weight.hpp"

#include <algorithm>

#include "llmi/model/kernels.hpp"

namespace llmi {

Weight Weight::from_float(std::vector<float> w) {
  Weight weight;
  weight.type_ = quant::Type::F32;
  weight.f32_ = std::move(w);
  return weight;
}

Weight Weight::quantized(const std::vector<float>& w, std::size_t out, std::size_t in, quant::Type type) {
  if (type == quant::Type::F32) return from_float(w);
  Weight weight;
  weight.type_ = type;
  weight.q_ = quant::quantize(w.data(), out, in, type);
  return weight;
}

bool Weight::empty() const { return type_ == quant::Type::F32 ? f32_.empty() : q_.empty(); }

std::size_t Weight::byte_size() const {
  return type_ == quant::Type::F32 ? f32_.size() * sizeof(float) : q_.byte_size();
}

void Weight::matmul(const float* x, std::size_t rows, std::size_t in, std::size_t out, const float* bias,
                    float* y) const {
  if (type_ == quant::Type::F32) {
    kernels::matmul(x, rows, in, f32_.data(), out, bias, y);
  } else {
    quant::matmul(x, rows, in, q_, out, bias, y);
  }
}

void Weight::row(std::size_t index, std::size_t in, float* dst) const {
  if (type_ == quant::Type::F32) {
    std::copy_n(f32_.data() + index * in, in, dst);
  } else {
    quant::dequantize_row(q_, index, dst);
  }
}

}  // namespace llmi
