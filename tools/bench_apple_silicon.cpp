// bench_apple_silicon: the M8 performance study's microbenchmarks -- tokenizer
// throughput, float32 vs Q8_0/Q4_0 matmul at the models' real decode shapes, a
// memory-bandwidth roofline, and the KV cache's read cost as context grows. No
// model weights: like tools/bench_kernels.cpp it uses random data at the real
// shapes, so it runs anywhere the engine compiles. Meant to be run on Apple
// Silicon (the arm64 NEON kernels) and on x86 (AVX2) for comparison.
//
//   g++ -O3 -std=c++20 -I include \
//       tools/bench_apple_silicon.cpp src/model/kernels.cpp src/model/quant.cpp \
//       src/util/thread_pool.cpp src/tokenizer/tokenizer.cpp src/util/json.cpp \
//       src/util/utf8.cpp -pthread -o bench_apple_silicon
//   ./bench_apple_silicon testdata/smollm2-135m/tokenizer.json
//
// On arm64 NEON is part of the base ISA (no flag). On x86 add -mavx2 -mfma to
// match the engine's CI build (LLMI_ENABLE_SIMD).

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "llmi/model/kernels.hpp"
#include "llmi/model/quant.hpp"
#include "llmi/tokenizer/tokenizer.hpp"
#include "llmi/util/result.hpp"

// tokenizer.cpp defines Tokenizer::load(), which references llmi::read_file
// (defined in model.cpp). This tool loads the tokenizer with from_json() and
// does not want model.cpp's dependencies, so it defines read_file itself.
namespace llmi {
Result<std::string> read_file(const std::string& path, std::size_t max_bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return fail("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  std::string s = ss.str();
  if (s.size() > max_bytes) return fail("file too large: " + path);
  return s;
}
}  // namespace llmi

namespace {

using Clock = std::chrono::steady_clock;
double secs(Clock::time_point a) { return std::chrono::duration<double>(Clock::now() - a).count(); }

std::vector<float> random_vec(std::size_t n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> d(-1.0F, 1.0F);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

// --- 1. Memory bandwidth: STREAM-style triad a = b + s*c over large arrays. ---
void bench_bandwidth() {
  std::printf("\n== 1. Memory bandwidth (STREAM triad a = b + s*c) ==\n");
  const std::size_t n = 32u * 1024u * 1024u;  // 32M floats per array = 128 MB x3
  std::vector<float> a(n), b(n), c(n);
  for (std::size_t i = 0; i < n; ++i) {
    b[i] = static_cast<float>(i % 7);
    c[i] = static_cast<float>(i % 5);
  }
  const float s = 3.0F;
  const int iters = 10;
  volatile float sink = 0.0F;
  const auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    for (std::size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
    sink = sink + a[it % n];
  }
  const double t = secs(t0);
  (void)sink;
  // 3 arrays touched per element (read b, read c, write a) x 4 bytes.
  const double gb = static_cast<double>(iters) * static_cast<double>(n) * 3.0 * 4.0 / 1e9;
  std::printf("  arrays %zu floats (%.0f MB each); %.1f GB moved in %.3f s => %.1f GB/s (single thread)\n", n,
              static_cast<double>(n) * 4.0 / 1e6, gb, t, gb / t);
}

// --- 2. Matmul at decode shapes: F32 vs Q8_0 vs Q4_0 (random weights). ---
struct Shape {
  const char* name;
  std::size_t in, out;
};

struct F32Result {
  double gbps, gflops;
};

F32Result bench_matmul_shape(const Shape& sh, int iters) {
  const std::vector<float> x = random_vec(sh.in, 1);
  const std::vector<float> w = random_vec(sh.out * sh.in, 2);
  std::vector<float> y(sh.out);
  const double flops = 2.0 * static_cast<double>(sh.in) * static_cast<double>(sh.out);

  struct Row {
    const char* label;
    double ms, gflops, gbps, mib;
  };
  std::vector<Row> rows;

  {  // F32
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) llmi::kernels::matmul(x.data(), 1, sh.in, w.data(), sh.out, nullptr, y.data());
    const double t = secs(t0) / iters;
    const double bytes = static_cast<double>(sh.out) * static_cast<double>(sh.in) * 4.0;
    rows.push_back({"f32  ", t * 1e3, flops / t / 1e9, bytes / t / 1e9, bytes / (1024.0 * 1024.0)});
  }
  for (const auto qt : {llmi::quant::Type::Q8_0, llmi::quant::Type::Q4_0}) {
    const auto qm = llmi::quant::quantize(w.data(), sh.out, sh.in, qt);
    const auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i) llmi::quant::matmul(x.data(), 1, sh.in, qm, sh.out, nullptr, y.data());
    const double t = secs(t0) / iters;
    const double bytes = static_cast<double>(qm.byte_size());
    rows.push_back({llmi::quant::type_name(qt).data(), t * 1e3, flops / t / 1e9, bytes / t / 1e9,
                    bytes / (1024.0 * 1024.0)});
  }

  std::printf("  %-26s", sh.name);
  std::printf("  (%zux%zu)\n", sh.out, sh.in);
  for (const auto& r : rows) {
    std::printf("      %-5s  %8.4f ms  %7.1f GFLOP/s  %7.1f GB/s  weight %7.2f MiB\n", r.label, r.ms, r.gflops,
                r.gbps, r.mib);
  }
  return {rows[0].gbps, rows[0].gflops};  // the f32 row
}

F32Result bench_matmul() {
  std::printf("\n== 2. Matmul at SmolLM2-135M decode shapes (rows=1), F32 vs quantized ==\n");
  std::printf("   (GB/s = weight bytes read / time; f32 should sit near the bandwidth ceiling, the quantized\n");
  std::printf("    kernels move fewer bytes but have no SIMD dequant yet, so they are compute-bound -- M5/ADR 0006)\n");
  const Shape shapes[] = {
      {"attn qkv/o  576x576", 576, 576},
      {"mlp up/gate 576x1536", 576, 1536},
      {"mlp down    1536x576", 1536, 576},
      {"lm_head     576x49152", 576, 49152},
  };
  F32Result lm_head{};
  for (const auto& sh : shapes) {
    const F32Result r = bench_matmul_shape(sh, sh.out >= 40000 ? 60 : 1500);
    if (sh.out >= 40000) lm_head = r;  // the cache-busting one
  }
  return lm_head;
}

// --- 3. Roofline: arithmetic intensity of decode matmul is ~0.5 FLOP/byte. ---
void roofline_note(double triad_gbps, F32Result lm_head) {
  std::printf("\n== 3. Roofline for single-token (rows=1) decode matmul ==\n");
  std::printf("   Each weight is read once and used for two flops (a multiply and an add): arithmetic intensity\n");
  std::printf("   2 flops / 4 bytes = 0.5 FLOP/byte for f32, far below any ridge point -- deeply memory bound.\n");
  std::printf("   The triad above (~%.0f GB/s) is copy bandwidth; the honest read-bandwidth anchor is the\n", triad_gbps);
  std::printf("   lm_head matmul, whose 108 MiB of weights cannot fit in cache: it ran at %.1f GB/s and\n", lm_head.gbps);
  std::printf("   %.1f GFLOP/s, and %.1f x 0.5 = %.1f, so it is reading weights at full bandwidth and the flops\n",
              lm_head.gflops, lm_head.gbps, lm_head.gbps * 0.5);
  std::printf("   come for free -- textbook memory-bound. The small per-layer matmuls above report higher GB/s\n");
  std::printf("   only because a few-MiB weight partly fits in cache. The lever is moving fewer weight bytes\n");
  std::printf("   (quantization); it pays off once its dequant is vectorized (section 2 shows it is not yet).\n");
}

// --- 4. KV cache: per-token attention read cost as context grows. ---
// SmolLM2-135M: 30 layers, num_kv_heads*head_dim = 192 (GQA), head_dim 64 ->
// 3 kv heads. Layout (transformer.hpp): k_[layer]/v_[layer] = capacity * 192,
// row-major by position. Each decode step reads all past K and V once.
double attention_pass(const float* k, const float* v, const float* q, std::size_t ctx, std::size_t layers,
                      std::size_t heads, std::size_t head_dim, std::size_t num_kv, bool head_major) {
  // Offset of KV head `kvh`, position `p`. position-major is the engine's
  // layout ([position][kv_head][head_dim]); head-major is [kv_head][position]
  // [head_dim], so one head's keys across positions are contiguous.
  const std::size_t kv = num_kv * head_dim;
  std::vector<float> scores(ctx), out(heads * head_dim);
  const int iters = 30;
  volatile double sink = 0.0;
  const auto t0 = Clock::now();
  for (int it = 0; it < iters; ++it) {
    for (std::size_t l = 0; l < layers; ++l) {
      (void)l;
      for (std::size_t h = 0; h < heads; ++h) {
        const std::size_t kvh = h / (heads / num_kv);  // GQA: query heads share a KV head
        const float* qh = q + h * head_dim;
        for (std::size_t p = 0; p < ctx; ++p) {
          const std::size_t off = head_major ? (kvh * ctx + p) * head_dim : p * kv + kvh * head_dim;
          scores[p] = llmi::kernels::dot(qh, k + off, head_dim);
        }
        llmi::kernels::softmax(scores.data(), ctx);
        float* oh = out.data() + h * head_dim;
        for (std::size_t j = 0; j < head_dim; ++j) oh[j] = 0.0F;
        for (std::size_t p = 0; p < ctx; ++p) {
          const std::size_t off = head_major ? (kvh * ctx + p) * head_dim : p * kv + kvh * head_dim;
          const float* vp = v + off;
          const float wt = scores[p];
          for (std::size_t j = 0; j < head_dim; ++j) oh[j] += wt * vp[j];
        }
      }
    }
    sink = sink + out[0];
  }
  (void)sink;
  return secs(t0) / iters;
}

void bench_kv() {
  std::printf("\n== 4. KV cache read cost per decode step, and the effect of its layout ==\n");
  std::printf("   SmolLM2-135M: 30 layers, 3 KV heads x 64 = 192 floats of K and 192 of V per position per\n");
  std::printf("   layer. A step reads all past K and V once. The engine stores them position-major\n");
  std::printf("   ([position][kv_head][head_dim]); head-major ([kv_head][position][head_dim]) makes one\n");
  std::printf("   head's keys contiguous across positions. Same data, same math, different layout:\n");
  constexpr std::size_t layers = 30, num_kv = 3, heads = 9, head_dim = 64;
  const std::size_t kv = num_kv * head_dim;
  std::mt19937_64 rng(7);
  std::uniform_real_distribution<float> d(-1.0F, 1.0F);
  const std::vector<float> q = random_vec(heads * head_dim, 9);
  for (const std::size_t ctx : {128u, 512u, 1024u, 2048u, 4096u}) {
    std::vector<float> k(ctx * kv), v(ctx * kv), k2(ctx * kv), v2(ctx * kv);
    for (auto& x : k) x = d(rng);
    for (auto& x : v) x = d(rng);
    // Reorder into head-major so both hold the same values.
    for (std::size_t p = 0; p < ctx; ++p)
      for (std::size_t kvh = 0; kvh < num_kv; ++kvh)
        for (std::size_t j = 0; j < head_dim; ++j) {
          k2[(kvh * ctx + p) * head_dim + j] = k[p * kv + kvh * head_dim + j];
          v2[(kvh * ctx + p) * head_dim + j] = v[p * kv + kvh * head_dim + j];
        }
    const double t_pos = attention_pass(k.data(), v.data(), q.data(), ctx, layers, heads, head_dim, num_kv, false);
    const double t_head = attention_pass(k2.data(), v2.data(), q.data(), ctx, layers, heads, head_dim, num_kv, true);
    const double bytes = static_cast<double>(layers) * static_cast<double>(ctx) * static_cast<double>(kv) * 2.0 * 4.0;
    std::printf("  ctx %5zu: KV read %6.1f MiB/token | position-major %7.3f ms (%5.1f GB/s) | head-major %7.3f ms "
                "(%5.1f GB/s)\n",
                ctx, bytes / (1024.0 * 1024.0), t_pos * 1e3, bytes / t_pos / 1e9, t_head * 1e3,
                bytes / t_head / 1e9);
  }
}

// --- Tokenizer throughput ---
void bench_tokenizer(const std::string& tok_path) {
  std::printf("\n== 5. Tokenizer throughput ==\n");
  std::ifstream in(tok_path, std::ios::binary);
  if (!in) {
    std::printf("  (skipped: cannot open %s)\n", tok_path.c_str());
    return;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  auto tok = llmi::Tokenizer::from_json(ss.str());
  if (!tok) {
    std::printf("  (skipped: %s)\n", tok.error().c_str());
    return;
  }
  // A varied ~4 MB corpus.
  const std::string unit =
      "The quick brown fox jumps over 13 lazy dogs, then writes JSON: {\"id\": 42, \"ok\": true}.\n"
      "Maintenance log: truck 12 at 84,000 miles needs brakes & an oil change. Visit https://example.com.\n"
      "Les na:ve re:sume: na:ve caf:: -- mixed Unicode, punctuation; and numbers 3.14159, 2718281828.\n";
  std::string corpus;
  corpus.reserve(4u * 1024u * 1024u);
  while (corpus.size() < 4u * 1024u * 1024u) corpus += unit;
  const double mb = static_cast<double>(corpus.size()) / 1e6;

  const auto t0 = Clock::now();
  auto ids = tok->encode(corpus, false);
  const double te = secs(t0);
  if (!ids) {
    std::printf("  (encode failed: %s)\n", ids.error().c_str());
    return;
  }
  const double ntok = static_cast<double>(ids->size());
  std::printf("  encode: %.2f MB -> %.0f tokens in %.3f s => %.1f MB/s, %.2f M tokens/s\n", mb, ntok, te, mb / te,
              ntok / te / 1e6);
  const auto t1 = Clock::now();
  auto back = tok->decode(*ids, false);
  const double td = secs(t1);
  if (back) {
    std::printf("  decode: %.0f tokens -> %.2f MB in %.3f s => %.2f M tokens/s%s\n", ntok,
                static_cast<double>(back->size()) / 1e6, td, ntok / td / 1e6,
                (*back == corpus) ? " (round-trip exact)" : " (round-trip DIFFERS)");
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string tok_path = argc > 1 ? argv[1] : "testdata/smollm2-135m/tokenizer.json";
#if defined(__aarch64__)
  std::printf("arch: aarch64 (NEON). ");
#elif defined(__x86_64__)
  std::printf("arch: x86_64. ");
#endif
  std::printf("A single-threaded study of where single-token decoding spends time.\n");

  // Measure bandwidth first so the roofline can quote it.
  const std::size_t n = 32u * 1024u * 1024u;
  std::vector<float> a(n), b(n), c(n);
  for (std::size_t i = 0; i < n; ++i) {
    b[i] = static_cast<float>(i % 7);
    c[i] = static_cast<float>(i % 5);
  }
  double peak = 0.0;
  {
    const auto t0 = Clock::now();
    for (int it = 0; it < 10; ++it)
      for (std::size_t i = 0; i < n; ++i) a[i] = b[i] + 3.0F * c[i];
    const double t = secs(t0);
    peak = 10.0 * static_cast<double>(n) * 3.0 * 4.0 / 1e9 / t;
  }

  bench_bandwidth();
  const F32Result lm = bench_matmul();
  roofline_note(peak, lm);
  bench_kv();
  bench_tokenizer(tok_path);
  return 0;
}
