#pragma once

#include <cstddef>
#include <map>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "llmi/model/safetensors.hpp"
#include "llmi/util/result.hpp"

namespace llmi {

// A model's weights as one lookup-by-name surface over one or more
// safetensors files. A single-file model (model.safetensors) and a sharded
// one (model.safetensors.index.json + model-0000k-of-....safetensors) are
// both read into a Weights, so everything downstream -- the Model checker and
// the Transformer -- queries tensors by name without caring how many files
// they came from.
//
// Each shard is parsed and validated on its own by SafetensorsFile (ADR 0001);
// Weights adds the one cross-file invariant a single file cannot have: a
// tensor name must not appear in two shards.
class Weights {
 public:
  Weights() = default;

  // One file. Cannot fail: a single shard has no cross-file conflict.
  static Weights single(SafetensorsFile file);
  // Several shards. Fails if any tensor name appears in more than one.
  static Result<Weights> sharded(std::vector<SafetensorsFile> shards);

  // The tensor with this name, or nullptr. Pointer stays valid for the life
  // of the Weights.
  [[nodiscard]] const TensorInfo* find(std::string_view name) const;
  // The tensor's raw bytes, from whichever shard holds it.
  [[nodiscard]] std::span<const std::byte> data(const TensorInfo& t) const;
  // Every tensor across every shard, sorted by name.
  [[nodiscard]] const std::vector<TensorInfo>& tensors() const { return all_; }
  [[nodiscard]] std::size_t shard_count() const { return shards_.size(); }

 private:
  // Builds all_ and owner_ from shards_, rejecting a name in two shards.
  static Result<Weights> build(std::vector<SafetensorsFile> shards);

  std::vector<SafetensorsFile> shards_;
  std::vector<TensorInfo> all_;                             // flattened, sorted by name
  std::map<std::string, std::size_t, std::less<>> owner_;   // tensor name -> index into shards_
};

}  // namespace llmi
