#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace llmi::util {

// A small fixed-size thread pool for parallel_for over independent,
// roughly-equal chunks of work (kernels::matmul's independent output rows).
// Worker threads are created once and reused for every call, so per-call
// overhead is a handful of atomic/condition-variable operations, not thread
// creation -- important since matmul runs this every layer, every token.
//
// Not a general task queue: parallel_for is the only entry point, it blocks
// the calling thread until every chunk has finished, and calling it
// concurrently from two threads (or from inside another parallel_for) is not
// supported.
class ThreadPool {
 public:
  // The process-wide pool, sized to std::thread::hardware_concurrency() - 1
  // worker threads (the calling thread itself does a share of the work), or
  // 0 worker threads if that can't be determined or is 1, in which case
  // parallel_for just runs everything on the calling thread. Created on
  // first use.
  static ThreadPool& shared();

  // num_workers additional worker threads, on top of whichever thread calls
  // parallel_for.
  explicit ThreadPool(std::size_t num_workers);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

  // Number of worker threads, not counting the calling thread.
  [[nodiscard]] std::size_t num_workers() const { return workers_.size(); }

  // Calls fn(begin, end) once per chunk, partitioning [0, n) into
  // num_workers() + 1 roughly-equal, non-overlapping chunks (the "+1" is the
  // calling thread's own share), then blocks until every chunk has returned.
  // If there are no worker threads, or n < min_total_work, runs fn(0, n)
  // directly on the calling thread instead of dispatching -- parallelizing
  // trivially small work costs more in synchronization than it saves.
  void parallel_for(std::size_t n, std::size_t min_total_work, const std::function<void(std::size_t, std::size_t)>& fn);

 private:
  void worker_loop(std::size_t worker_index);

  std::vector<std::thread> workers_;

  std::mutex mu_;
  std::condition_variable job_ready_;
  std::condition_variable job_done_;
  const std::function<void(std::size_t, std::size_t)>* fn_ = nullptr;
  std::vector<std::pair<std::size_t, std::size_t>> ranges_;  // one [begin,end) per worker, index 0..num_workers()-1
  std::size_t version_ = 0;  // bumped each dispatch; each worker remembers the version it last ran
  std::size_t pending_ = 0;  // workers still running this dispatch
  bool stop_ = false;
};

}  // namespace llmi::util
