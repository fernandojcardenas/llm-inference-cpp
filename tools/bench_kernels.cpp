// bench_kernels: microbenchmarks kernels::matmul at this project's real
// per-layer shapes (SmolLM2-135M and Qwen2.5-0.5B, single-token decode), to
// see where time actually goes before optimizing anything -- the profiling
// step behind M4 (docs/performance.md, docs/evidence/m4-profile.txt). Not
// part of the normal build; compile and run directly, e.g. against the
// current library sources:
//
//   g++ -O3 -std=c++20 -mavx2 -mfma -I include \
//       tools/bench_kernels.cpp src/model/kernels.cpp src/util/thread_pool.cpp \
//       -pthread -o bench_kernels && ./bench_kernels
//
// (Drop -mavx2 -mfma, or add -march=armv8-a+simd / nothing at all on arm64,
// to see the portable-scalar or NEON path respectively -- see kernels.cpp's
// compile-time dispatch.)

#include "llmi/model/kernels.hpp"
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using clk = std::chrono::steady_clock;

double bench_matmul(std::size_t rows, std::size_t in, std::size_t out, int iters) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
  std::vector<float> x(rows * in), w(out * in), y(rows * out);
  for (auto& v : x) v = dist(rng);
  for (auto& v : w) v = dist(rng);
  auto t0 = clk::now();
  for (int i = 0; i < iters; ++i) llmi::kernels::matmul(x.data(), rows, in, w.data(), out, nullptr, y.data());
  auto t1 = clk::now();
  double secs = std::chrono::duration<double>(t1 - t0).count();
  double macs = static_cast<double>(rows) * in * out * iters;
  printf("matmul rows=%zu in=%zu out=%zu: %d iters, %.4f s total, %.4f ms/call, %.2f GFLOP/s\n",
         rows, in, out, iters, secs, secs / iters * 1000.0, macs * 2.0 / secs / 1e9);
  return secs / iters;
}

int main() {
  printf("-- SmolLM2-135M shapes (H=576, F=1536, nh*hd=576, nkv*hd=192), decode (rows=1) --\n");
  double wq = bench_matmul(1, 576, 576, 2000);
  double wkv = bench_matmul(1, 576, 192, 2000);
  double wo = bench_matmul(1, 576, 576, 2000);
  double gate = bench_matmul(1, 576, 1536, 2000);
  double down = bench_matmul(1, 1536, 576, 2000);
  double logits_smol = bench_matmul(1, 576, 49152, 50);
  double per_layer = wq + 2 * wkv + wo + 2 * gate + down;
  printf("  => ~per-layer %.4f ms x 30 layers = %.2f ms, + logits %.2f ms => ~%.2f ms/token total\n",
         per_layer * 1000, per_layer * 1000 * 30, logits_smol * 1000, per_layer * 1000 * 30 + logits_smol * 1000);

  printf("\n-- Qwen2.5-0.5B shapes (H=896, F=4864, nh*hd=896, nkv*hd=128), decode (rows=1) --\n");
  double q_wq = bench_matmul(1, 896, 896, 2000);
  double q_wkv = bench_matmul(1, 896, 128, 2000);
  double q_wo = bench_matmul(1, 896, 896, 2000);
  double q_gate = bench_matmul(1, 896, 4864, 1000);
  double q_down = bench_matmul(1, 4864, 896, 1000);
  double q_logits = bench_matmul(1, 896, 151936, 20);
  double q_per_layer = q_wq + 2 * q_wkv + q_wo + 2 * q_gate + q_down;
  printf("  => ~per-layer %.4f ms x 24 layers = %.2f ms, + logits %.2f ms => ~%.2f ms/token total\n",
         q_per_layer * 1000, q_per_layer * 1000 * 24, q_logits * 1000, q_per_layer * 1000 * 24 + q_logits * 1000);
  return 0;
}
