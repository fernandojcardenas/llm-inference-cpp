#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "llmi/util/result.hpp"

namespace llmi {

// Limits applied while reading a safetensors index; defaults are generous
// for real models (the largest open checkpoints ship a few hundred shards)
// and low enough that nothing here can overflow.
struct SafetensorsIndexLimits {
  std::uint64_t max_index_bytes = 64ULL * 1024 * 1024;  // the index JSON itself
  std::size_t max_entries = 2'000'000;                  // tensor -> shard entries
  std::size_t max_shards = 100'000;                     // distinct shard files
  std::size_t max_name_bytes = 1024;                    // a tensor name
  std::size_t max_filename_bytes = 255;                 // a shard filename
};

// A parsed Hugging Face safetensors index (model.safetensors.index.json):
//
//   { "metadata": { ... }, "weight_map": { "<tensor>": "<shard file>", ... } }
//
// It names the shard file that holds each tensor of a model too large for a
// single file. Like every other file this engine reads, the index is treated
// as untrusted (ADR 0001): the shard filenames in it are used to open files
// next to the index, so parsing rejects anything that is not a plain filename
// in that one directory -- no path separators, no "." or "..", no leading dot,
// and a ".safetensors" suffix -- which closes path traversal before a single
// byte is opened. Unknown top-level keys, a missing or empty weight_map, a
// non-string filename, and oversized inputs are all errors, not warnings.
class SafetensorsIndex {
 public:
  static Result<SafetensorsIndex> parse(std::string_view json_text, const SafetensorsIndexLimits& limits = {});

  // Tensor name -> shard filename, in name order.
  [[nodiscard]] const std::map<std::string, std::string, std::less<>>& weight_map() const { return weight_map_; }
  // The distinct shard filenames, sorted and de-duplicated. Each is a plain,
  // validated filename safe to append to the index's own directory.
  [[nodiscard]] const std::vector<std::string>& shards() const { return shards_; }

 private:
  std::map<std::string, std::string, std::less<>> weight_map_;
  std::vector<std::string> shards_;
};

// True if `name` is a plain filename safe to open next to the index: non-empty,
// within `max_bytes`, made only of [A-Za-z0-9._-], not "." or "..", not
// starting with '.' or '-', and ending in ".safetensors". Exposed for testing.
bool is_safe_shard_filename(std::string_view name, std::size_t max_bytes);

}  // namespace llmi
