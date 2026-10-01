#include "llmi/util/thread_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <numeric>
#include <vector>

using llmi::util::ThreadPool;

namespace {

// Runs parallel_for over [0, n) and returns, for every index, how many times
// a chunk covered it -- correctness means every entry is exactly 1: the
// chunks partition the range with no gaps and no overlaps, however many
// worker threads actually ran.
std::vector<int> coverage(ThreadPool& pool, std::size_t n, std::size_t min_total_work) {
  std::vector<std::atomic<int>> hits(n);
  pool.parallel_for(n, min_total_work, [&](std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; ++i) ++hits[i];
  });
  std::vector<int> out(n);
  for (std::size_t i = 0; i < n; ++i) out[i] = hits[i].load();
  return out;
}

}  // namespace

TEST(ThreadPool, CoversEveryIndexExactlyOnce) {
  ThreadPool& pool = ThreadPool::shared();
  for (std::size_t n : {0U, 1U, 2U, 3U, 7U, 100U, 10000U}) {
    // min_total_work 0 forces dispatch whenever there are worker threads, so
    // this exercises the multi-chunk path on any machine, including this
    // project's 2-core CI runners.
    auto cov = coverage(pool, n, 0);
    ASSERT_EQ(cov.size(), n);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(cov[i], 1) << "n=" << n << " i=" << i;
  }
}

TEST(ThreadPool, SmallWorkStaysOnTheCallingThreadAndIsStillCorrect) {
  ThreadPool& pool = ThreadPool::shared();
  // A min_total_work far larger than n always takes the "run inline" branch.
  auto cov = coverage(pool, 50, 1'000'000);
  for (int c : cov) EXPECT_EQ(c, 1);
}

TEST(ThreadPool, ConstructedDirectlyWithKnownWorkerCount) {
  // A pool built with an explicit worker count, independent of this
  // machine's hardware_concurrency(), so the chunk math itself is checked
  // regardless of how many cores CI happens to run on.
  ThreadPool pool(3);
  EXPECT_EQ(pool.num_workers(), 3U);
  for (std::size_t n : {0U, 1U, 4U, 5U, 97U}) {
    auto cov = coverage(pool, n, 0);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(cov[i], 1) << "n=" << n << " i=" << i;
  }
}

TEST(ThreadPool, ZeroWorkerPoolRunsEverythingInline) {
  ThreadPool pool(0);
  EXPECT_EQ(pool.num_workers(), 0U);
  auto cov = coverage(pool, 1000, 0);
  for (int c : cov) EXPECT_EQ(c, 1);
}

TEST(ThreadPool, ConsecutiveCallsDontInterfereWithEachOther) {
  // Nothing from one dispatch (the version counter, stale ranges) should
  // leak into the next call.
  ThreadPool pool(2);
  for (int iter = 0; iter < 20; ++iter) {
    auto cov = coverage(pool, 123, 0);
    for (int c : cov) ASSERT_EQ(c, 1);
  }
}
