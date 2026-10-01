#include "llmi/util/thread_pool.hpp"

#include <algorithm>

namespace llmi::util {

ThreadPool& ThreadPool::shared() {
  const unsigned hw = std::thread::hardware_concurrency();
  // hardware_concurrency() can return 0 if it can't be determined. One
  // worker thread plus the calling thread (2-way) is a safe default; a
  // single-core machine still works correctly, just without a speed-up.
  const std::size_t workers = hw > 1 ? static_cast<std::size_t>(hw) - 1 : 0;
  static ThreadPool pool(workers);
  return pool;
}

ThreadPool::ThreadPool(std::size_t num_workers) {
  workers_.reserve(num_workers);
  for (std::size_t i = 0; i < num_workers; ++i) {
    workers_.emplace_back([this, i] { worker_loop(i); });
  }
}

ThreadPool::~ThreadPool() {
  {
    const std::lock_guard<std::mutex> lock(mu_);
    stop_ = true;
  }
  job_ready_.notify_all();
  for (auto& t : workers_) t.join();
}

void ThreadPool::worker_loop(std::size_t worker_index) {
  std::size_t last_version = 0;
  while (true) {
    std::pair<std::size_t, std::size_t> range;
    const std::function<void(std::size_t, std::size_t)>* fn = nullptr;
    {
      std::unique_lock<std::mutex> lock(mu_);
      job_ready_.wait(lock, [&] { return stop_ || version_ != last_version; });
      if (stop_) return;
      last_version = version_;
      range = ranges_[worker_index];
      fn = fn_;
    }
    (*fn)(range.first, range.second);
    {
      const std::lock_guard<std::mutex> lock(mu_);
      --pending_;
      if (pending_ == 0) job_done_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(std::size_t n, std::size_t min_total_work,
                               const std::function<void(std::size_t, std::size_t)>& fn) {
  if (n == 0) return;
  if (workers_.empty() || n < min_total_work) {
    fn(0, n);
    return;
  }

  const std::size_t slots = workers_.size() + 1;  // worker threads + the calling thread
  const std::size_t chunk = (n + slots - 1) / slots;
  std::vector<std::pair<std::size_t, std::size_t>> ranges(slots);
  for (std::size_t s = 0; s < slots; ++s) {
    ranges[s] = {std::min(s * chunk, n), std::min((s + 1) * chunk, n)};
  }

  {
    const std::lock_guard<std::mutex> lock(mu_);
    fn_ = &fn;
    ranges_.assign(ranges.begin(), ranges.begin() + static_cast<std::ptrdiff_t>(workers_.size()));
    pending_ = workers_.size();
    ++version_;
  }
  job_ready_.notify_all();

  // The calling thread does the last slot itself, instead of idling.
  const auto own = ranges[workers_.size()];
  fn(own.first, own.second);

  std::unique_lock<std::mutex> lock(mu_);
  job_done_.wait(lock, [&] { return pending_ == 0; });
}

}  // namespace llmi::util
