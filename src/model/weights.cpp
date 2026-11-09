#include "llmi/model/weights.hpp"

#include <algorithm>
#include <utility>

namespace llmi {

Result<Weights> Weights::build(std::vector<SafetensorsFile> shards) {
  Weights w;
  w.shards_ = std::move(shards);
  for (std::size_t i = 0; i < w.shards_.size(); ++i) {
    for (const auto& t : w.shards_[i].tensors()) {
      auto [it, inserted] = w.owner_.emplace(t.name, i);
      if (!inserted) return fail("weights: tensor \"" + t.name + "\" appears in more than one shard");
      w.all_.push_back(t);
    }
  }
  std::sort(w.all_.begin(), w.all_.end(), [](const TensorInfo& a, const TensorInfo& b) { return a.name < b.name; });
  return w;
}

Weights Weights::single(SafetensorsFile file) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(std::move(file));
  // One shard cannot collide with itself (names within a file are unique), so
  // build() cannot fail here.
  return std::move(build(std::move(shards)).value());
}

Result<Weights> Weights::sharded(std::vector<SafetensorsFile> shards) { return build(std::move(shards)); }

const TensorInfo* Weights::find(std::string_view name) const {
  auto it = std::lower_bound(all_.begin(), all_.end(), name,
                             [](const TensorInfo& t, std::string_view n) { return t.name < n; });
  if (it == all_.end() || it->name != name) return nullptr;
  return &*it;
}

std::span<const std::byte> Weights::data(const TensorInfo& t) const {
  auto it = owner_.find(t.name);
  if (it == owner_.end()) return {};
  return shards_[it->second].data(t);
}

}  // namespace llmi
