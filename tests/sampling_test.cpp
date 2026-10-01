#include "llmi/model/sampling.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>

using namespace llmi;

namespace {
double sum(const std::vector<float>& v) { return std::accumulate(v.begin(), v.end(), 0.0); }
}  // namespace

TEST(Sampling, TemperatureZeroIsGreedy) {
  const float logits[] = {1, 5, 3, 5, 2};  // two tied maxima at index 1 and 3
  SamplingConfig cfg;
  cfg.temperature = 0.0F;
  std::mt19937_64 rng(1);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(sample(logits, 5, cfg, rng), 1);  // first maximum, every draw
}

TEST(Sampling, ProbabilitiesSumToOneAndFollowSoftmax) {
  const float logits[] = {1, 2, 3, 4};
  SamplingConfig cfg;  // no filtering
  auto p = sampling_probabilities(logits, 4, cfg);
  EXPECT_NEAR(sum(p), 1.0, 1e-6);
  for (std::size_t i = 0; i + 1 < 4; ++i) EXPECT_LT(p[i], p[i + 1]);  // monotonic with logits
  // Matches a plain double-precision softmax.
  double denom = 0;
  for (float x : logits) denom += std::exp(static_cast<double>(x));
  for (std::size_t i = 0; i < 4; ++i) EXPECT_NEAR(p[i], std::exp(static_cast<double>(logits[i])) / denom, 1e-5);
}

TEST(Sampling, TemperatureFlattensOrSharpensTheDistribution) {
  const float logits[] = {0, 1, 2};
  SamplingConfig hot, cold;
  hot.temperature = 100.0F;   // near-uniform
  cold.temperature = 0.01F;   // near one-hot on the argmax
  auto ph = sampling_probabilities(logits, 3, hot);
  auto pc = sampling_probabilities(logits, 3, cold);
  EXPECT_NEAR(ph[0], ph[2], 0.05);  // hot: probabilities nearly equal
  EXPECT_GT(pc[2], 0.99F);          // cold: almost all mass on the best token
}

TEST(Sampling, TopKKeepsOnlyTheKHighest) {
  const float logits[] = {5, 1, 4, 2, 3};  // ranks: idx0=5(best) idx2=4 idx4=3 idx3=2 idx1=1
  SamplingConfig cfg;
  cfg.top_k = 2;
  auto p = sampling_probabilities(logits, 5, cfg);
  EXPECT_GT(p[0], 0.0F);
  EXPECT_GT(p[2], 0.0F);
  EXPECT_EQ(p[1], 0.0F);
  EXPECT_EQ(p[3], 0.0F);
  EXPECT_EQ(p[4], 0.0F);
  EXPECT_NEAR(sum(p), 1.0, 1e-6);
}

TEST(Sampling, TopPKeepsTheSmallestSufficientNucleus) {
  // One dominant token (~0.9 probability after softmax-like spacing) plus a long tail.
  const float logits[] = {10, 0, 0, 0, 0};
  SamplingConfig cfg;
  cfg.top_p = 0.5F;  // should keep just the dominant token
  auto p = sampling_probabilities(logits, 5, cfg);
  EXPECT_NEAR(p[0], 1.0F, 1e-5);
  for (std::size_t i = 1; i < 5; ++i) EXPECT_EQ(p[i], 0.0F);
}

TEST(Sampling, TopPAlwaysKeepsAtLeastOneToken) {
  const float logits[] = {1, 1, 1, 1};  // uniform: even top_p=0 must keep the (first) argmax
  SamplingConfig cfg;
  cfg.top_p = 0.0F;
  auto p = sampling_probabilities(logits, 4, cfg);
  EXPECT_NEAR(sum(p), 1.0, 1e-6);
  EXPECT_EQ(std::count_if(p.begin(), p.end(), [](float x) { return x > 0; }), 1);
}

TEST(Sampling, SameSeedReproducesTheSameDraws) {
  const float logits[] = {1, 2, 3, 2, 1};
  SamplingConfig cfg;
  cfg.temperature = 1.0F;
  std::mt19937_64 a(42), b(42);
  std::vector<TokenId> da, db;
  for (int i = 0; i < 20; ++i) {
    da.push_back(sample(logits, 5, cfg, a));
    db.push_back(sample(logits, 5, cfg, b));
  }
  EXPECT_EQ(da, db);
}

TEST(Sampling, DifferentSeedsCanDiffer) {
  const float logits[] = {1, 2, 3, 2, 1};
  SamplingConfig cfg;
  cfg.temperature = 1.0F;
  std::mt19937_64 a(1), b(2);
  bool any_different = false;
  for (int i = 0; i < 50; ++i) any_different = any_different || sample(logits, 5, cfg, a) != sample(logits, 5, cfg, b);
  EXPECT_TRUE(any_different);
}

TEST(Sampling, DrawsStayWithinTheFilteredSet) {
  const float logits[] = {5, 1, 4, 2, 3};
  SamplingConfig cfg;
  cfg.top_k = 2;  // only indices 0 and 2 may be drawn
  std::mt19937_64 rng(7);
  for (int i = 0; i < 200; ++i) {
    const TokenId t = sample(logits, 5, cfg, rng);
    EXPECT_TRUE(t == 0 || t == 2) << t;
  }
}
