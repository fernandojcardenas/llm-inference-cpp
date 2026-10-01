#pragma once

#include <cstddef>

namespace llmi::kernels {

// The numerical building blocks of the forward pass, kept free of model
// structure so each can be tested against a straightforward double-precision
// version. Plain C++ for now; M4 replaces the hot ones with threaded SIMD.

// Dot product of two float vectors, accumulated in 8 independent lanes so
// the compiler can vectorize it without -ffast-math.
float dot(const float* a, const float* b, std::size_t n);

// y[r][o] = sum_i x[r][i] * w[o][i] (+ bias[o]) for every row r.
// x is rows x in, w is out x in (the layout PyTorch stores Linear weights in),
// y is rows x out. bias may be null.
void matmul(const float* x, std::size_t rows, std::size_t in, const float* w, std::size_t out, const float* bias,
            float* y);

// y = x / sqrt(mean(x^2) + eps) * weight, over n values (RMSNorm).
void rmsnorm(const float* x, const float* weight, std::size_t n, float eps, float* y);

// Rotary position embedding on one head vector, "rotate half" layout: value
// i is paired with i + d/2 and rotated by pos * inv_freq[i].
void rope(float* v, std::size_t head_dim, std::size_t pos, const float* inv_freq);

// In-place softmax over n values.
void softmax(float* x, std::size_t n);

// x * sigmoid(x)
float silu(float x);

}  // namespace llmi::kernels
