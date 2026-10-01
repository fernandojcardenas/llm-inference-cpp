#include "llmi/model/transformer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>

#include "llmi/model/kernels.hpp"

using namespace llmi;

// ---------------------------------------------------------------- kernels

TEST(Kernels, DotMatchesDoublePrecision) {
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1, 1);
  for (std::size_t n : {0U, 1U, 7U, 8U, 9U, 64U, 577U}) {
    std::vector<float> a(n), b(n);
    double ref = 0;
    for (std::size_t i = 0; i < n; ++i) {
      a[i] = u(rng);
      b[i] = u(rng);
      ref += static_cast<double>(a[i]) * b[i];
    }
    EXPECT_NEAR(kernels::dot(a.data(), b.data(), n), ref, 1e-5) << n;
  }
}

TEST(Kernels, MatmulWithBias) {
  // x: 2 rows x 3, w: 2 outputs x 3
  const float x[] = {1, 2, 3, 4, 5, 6};
  const float w[] = {1, 0, -1, 0.5F, 0.5F, 0.5F};
  const float bias[] = {10, -1};
  float y[4];
  kernels::matmul(x, 2, 3, w, 2, bias, y);
  EXPECT_FLOAT_EQ(y[0], 1 - 3 + 10.0F);
  EXPECT_FLOAT_EQ(y[1], 3 - 1.0F);
  EXPECT_FLOAT_EQ(y[2], 4 - 6 + 10.0F);
  EXPECT_FLOAT_EQ(y[3], 7.5F - 1);
  kernels::matmul(x, 2, 3, w, 2, nullptr, y);
  EXPECT_FLOAT_EQ(y[0], -2.0F);
}

TEST(Kernels, RmsNorm) {
  const float x[] = {3, 4};
  const float w[] = {1, 2};
  float y[2];
  kernels::rmsnorm(x, w, 2, 0.0F, y);
  const double rms = std::sqrt((9.0 + 16.0) / 2.0);
  EXPECT_NEAR(y[0], 3 / rms, 1e-6);
  EXPECT_NEAR(y[1], 2 * 4 / rms, 1e-6);
  const float zeros[] = {0, 0};
  kernels::rmsnorm(zeros, w, 2, 1e-5F, y);  // epsilon keeps it finite
  EXPECT_EQ(y[0], 0.0F);
}

TEST(Kernels, RopeRotatesPairsAndKeepsLength) {
  const std::size_t d = 8;
  std::vector<float> inv(d / 2);
  for (std::size_t i = 0; i < d / 2; ++i) inv[i] = 1.0F / std::pow(10000.0F, static_cast<float>(2 * i) / d);
  std::vector<float> v = {1, 2, 3, 4, 5, 6, 7, 8};
  auto at0 = v;
  kernels::rope(at0.data(), d, 0, inv.data());
  EXPECT_EQ(at0, v);  // position 0: no rotation
  auto at3 = v;
  kernels::rope(at3.data(), d, 3, inv.data());
  // Pair (0, 4) rotated by 3 rad.
  EXPECT_NEAR(at3[0], 1 * std::cos(3.0) - 5 * std::sin(3.0), 1e-5);
  EXPECT_NEAR(at3[4], 5 * std::cos(3.0) + 1 * std::sin(3.0), 1e-5);
  EXPECT_NEAR(kernels::dot(at3.data(), at3.data(), d), kernels::dot(v.data(), v.data(), d), 1e-3);
}

TEST(Kernels, RopeScoresDependOnlyOnRelativePosition) {
  const std::size_t d = 16;
  std::vector<float> inv(d / 2);
  for (std::size_t i = 0; i < d / 2; ++i) inv[i] = 1.0F / std::pow(10000.0F, static_cast<float>(2 * i) / d);
  std::mt19937 rng(2);
  std::uniform_real_distribution<float> u(-1, 1);
  std::vector<float> q(d), k(d);
  for (auto& x : q) x = u(rng);
  for (auto& x : k) x = u(rng);
  auto score = [&](std::size_t m, std::size_t n) {
    auto a = q;
    auto b = k;
    kernels::rope(a.data(), d, m, inv.data());
    kernels::rope(b.data(), d, n, inv.data());
    return kernels::dot(a.data(), b.data(), d);
  };
  EXPECT_NEAR(score(5, 2), score(13, 10), 1e-4);
  EXPECT_NEAR(score(7, 7), score(0, 0), 1e-4);
}

TEST(Kernels, SoftmaxIsStableAndNormalised) {
  float x[] = {1000, 1001, 999};
  kernels::softmax(x, 3);
  EXPECT_NEAR(x[0] + x[1] + x[2], 1.0, 1e-6);
  EXPECT_GT(x[1], x[0]);
  EXPECT_NEAR(x[1] / x[0], std::exp(1.0), 1e-4);
  float one[] = {-5};
  kernels::softmax(one, 1);
  EXPECT_EQ(one[0], 1.0F);
}

TEST(Kernels, Silu) {
  EXPECT_EQ(kernels::silu(0), 0.0F);
  EXPECT_NEAR(kernels::silu(1), 1.0 / (1.0 + std::exp(-1.0)), 1e-6);
  EXPECT_NEAR(kernels::silu(30), 30.0F, 1e-5);
  EXPECT_NEAR(kernels::silu(-30), 0.0F, 1e-5);
}

TEST(Kernels, ArgmaxTakesTheFirstMaximum) {
  const float x[] = {1, 3, 2, 3};
  EXPECT_EQ(argmax(x, 4), 1U);
}

// ---------------------------------------------------------------- a tiny random model

namespace {

const char* kTiny = R"({
  "architectures": ["%s"], "vocab_size": 32, "hidden_size": 16, "intermediate_size": 24,
  "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 64,
  "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": %s, "eos_token_id": 3
})";

// Builds a model with random F32 weights in memory.
struct TinyModel {
  std::vector<std::byte> bytes;
  Model model;
};

TinyModel make_tiny(const char* arch, bool tied, unsigned seed) {
  char cfg_text[1024];
  std::snprintf(cfg_text, sizeof cfg_text, kTiny, arch, tied ? "true" : "false");
  auto cfg = parse_config(cfg_text);
  if (!cfg) throw std::runtime_error(cfg.error());
  const auto tensors = expected_tensors(*cfg);
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0F, 0.3F);
  std::string header = "{";
  std::vector<float> data;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    std::uint64_t n = 1;
    std::string shape;
    for (std::size_t d = 0; d < tensors[i].shape.size(); ++d) {
      n *= tensors[i].shape[d];
      shape += (d != 0 ? "," : "") + std::to_string(tensors[i].shape[d]);
    }
    const bool norm = tensors[i].name.find("norm") != std::string::npos;
    const std::uint64_t off = data.size() * 4;
    for (std::uint64_t k = 0; k < n; ++k) data.push_back(norm ? 1.0F + 0.1F * nd(rng) : nd(rng));
    header += (i != 0 ? "," : "") + std::string("\"") + tensors[i].name + "\":{\"dtype\":\"F32\",\"shape\":[" + shape +
              "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + n * 4) + "]}";
  }
  header += "}";
  TinyModel t;
  t.bytes.resize(8 + header.size() + data.size() * 4);
  for (int k = 0; k < 8; ++k) t.bytes[static_cast<std::size_t>(k)] = std::byte((header.size() >> (8U * static_cast<unsigned>(k))) & 0xFFU);
  std::memcpy(t.bytes.data() + 8, header.data(), header.size());
  std::memcpy(t.bytes.data() + 8 + header.size(), data.data(), data.size() * 4);
  auto st = SafetensorsFile::parse(t.bytes);
  if (!st) throw std::runtime_error(st.error());
  auto m = Model::from_parts(*cfg, std::move(st.value()));
  if (!m) throw std::runtime_error(m.error());
  t.model = std::move(m.value());
  return t;
}

}  // namespace

TEST(Transformer, IsCausal) {
  for (const char* arch : {"LlamaForCausalLM", "Qwen2ForCausalLM"}) {
    auto tiny = make_tiny(arch, true, 3);
    auto tf = Transformer::load(tiny.model);
    ASSERT_TRUE(tf);
    const std::vector<TokenId> a = {5, 9, 1, 30, 7};
    std::vector<TokenId> b = a;
    b[3] = 2;  // change a later token
    auto la = tf->forward(a, true);
    auto lb = tf->forward(b, true);
    ASSERT_TRUE(la && lb);
    const std::size_t V = 32;
    for (std::size_t i = 0; i < 3 * V; ++i) ASSERT_EQ((*la)[i], (*lb)[i]) << arch << " position " << i / V;
    bool differs = false;
    for (std::size_t i = 3 * V; i < 5 * V; ++i) differs = differs || (*la)[i] != (*lb)[i];
    EXPECT_TRUE(differs) << arch;
  }
}

TEST(Transformer, LastPositionMatchesAllPositions) {
  auto tiny = make_tiny("LlamaForCausalLM", false, 4);
  auto tf = Transformer::load(tiny.model);
  ASSERT_TRUE(tf);
  const std::vector<TokenId> ids = {1, 2, 3, 4};
  auto all = tf->forward(ids, true);
  auto last = tf->forward(ids, false);
  ASSERT_TRUE(all && last);
  ASSERT_EQ(last->size(), 32U);
  for (std::size_t i = 0; i < 32; ++i) EXPECT_EQ((*last)[i], (*all)[3 * 32 + i]);
}

TEST(Transformer, TracesEveryLayer) {
  auto tiny = make_tiny("Qwen2ForCausalLM", false, 5);
  auto tf = Transformer::load(tiny.model);
  ASSERT_TRUE(tf);
  ForwardTrace trace;
  ASSERT_TRUE(tf->forward({7, 8, 9}, false, &trace));
  ASSERT_EQ(trace.states.size(), 2U + 2U);  // embeddings, 2 layers, final norm
  for (const auto& s : trace.states) EXPECT_EQ(s.size(), 3U * 16U);
}

TEST(Transformer, RejectsBadInput) {
  auto tiny = make_tiny("LlamaForCausalLM", true, 6);
  auto tf = Transformer::load(tiny.model);
  ASSERT_TRUE(tf);
  EXPECT_FALSE(tf->forward({}));
  EXPECT_FALSE(tf->forward({32}));
  EXPECT_FALSE(tf->forward({-1}));
  EXPECT_FALSE(tf->forward(std::vector<TokenId>(65, 1)));  // longer than the context
}

TEST(Transformer, GreedyIsDeterministicAndStops) {
  auto tiny = make_tiny("LlamaForCausalLM", true, 7);
  auto tf = Transformer::load(tiny.model);
  ASSERT_TRUE(tf);
  auto a = tf->generate_greedy({1, 2}, 10, {});
  auto b = tf->generate_greedy({1, 2}, 10, {});
  ASSERT_TRUE(a && b);
  EXPECT_EQ(a->size(), 10U);
  EXPECT_EQ(a.value(), b.value());
  // Stop on the first generated token: generation ends right after it.
  auto stopped = tf->generate_greedy({1, 2}, 10, {(*a)[0]});
  ASSERT_TRUE(stopped);
  EXPECT_EQ(stopped->size(), 1U);
  EXPECT_EQ(tf->config().eos_token_ids, (std::vector<std::int32_t>{3}));
}
