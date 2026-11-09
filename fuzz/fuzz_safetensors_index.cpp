// Arbitrary bytes as a safetensors index (model.safetensors.index.json).
// Invariants: no crash or sanitizer report; in an accepted index every shard
// filename is a safe plain filename, weight_map is non-empty, and shards() is
// exactly the sorted set of distinct weight_map values.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <set>
#include <string>
#include <string_view>

#include "llmi/model/safetensors_index.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  const llmi::SafetensorsIndexLimits limits;
  auto idx = llmi::SafetensorsIndex::parse(text, limits);
  if (!idx) return 0;

  if (idx->weight_map().empty()) std::abort();

  std::set<std::string> from_map;
  for (const auto& [tensor, shard] : idx->weight_map()) {
    if (!llmi::is_safe_shard_filename(shard, limits.max_filename_bytes)) std::abort();
    from_map.insert(shard);
  }
  // shards() must be exactly the distinct, sorted filenames of the weight_map.
  if (idx->shards().size() != from_map.size()) std::abort();
  if (!std::equal(idx->shards().begin(), idx->shards().end(), from_map.begin())) std::abort();
  return 0;
}
