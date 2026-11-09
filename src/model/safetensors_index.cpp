#include "llmi/model/safetensors_index.hpp"

#include <set>

#include "llmi/util/json.hpp"

namespace llmi {

bool is_safe_shard_filename(std::string_view name, std::size_t max_bytes) {
  if (name.empty() || name.size() > max_bytes) return false;
  if (name == "." || name == "..") return false;
  // No path separators, drive letters or NULs: the name must stay in the
  // index's own directory. A leading '.' (hidden / "..") or '-' (so the name
  // can never be read as a command-line option) is refused up front.
  if (name.front() == '.' || name.front() == '-') return false;
  for (const char c : name) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' ||
                    c == '_' || c == '-';
    if (!ok) return false;
  }
  constexpr std::string_view kSuffix = ".safetensors";
  if (name.size() <= kSuffix.size()) return false;
  return name.substr(name.size() - kSuffix.size()) == kSuffix;
}

Result<SafetensorsIndex> SafetensorsIndex::parse(std::string_view json_text, const SafetensorsIndexLimits& limits) {
  if (json_text.size() > limits.max_index_bytes) {
    return fail("safetensors index: file larger than " + std::to_string(limits.max_index_bytes) + " bytes");
  }
  auto root = json::parse(json_text);
  if (!root) return fail("safetensors index: " + root.error());
  if (!root->is_object()) return fail("safetensors index: not a JSON object");

  // Exactly the two keys Hugging Face writes; anything else means the file is
  // not the index it claims to be (same strictness as the safetensors header).
  for (const auto& [key, value] : root->members()) {
    if (key != "metadata" && key != "weight_map") {
      return fail("safetensors index: unexpected key \"" + key + "\"");
    }
  }
  if (const json::Value* meta = root->find("metadata"); meta != nullptr && !meta->is_null() && !meta->is_object()) {
    return fail("safetensors index: \"metadata\" is not an object");
  }

  const json::Value* wm = root->find("weight_map");
  if (wm == nullptr || !wm->is_object()) return fail("safetensors index: missing \"weight_map\" object");
  if (wm->members().empty()) return fail("safetensors index: \"weight_map\" is empty");
  if (wm->members().size() > limits.max_entries) return fail("safetensors index: too many weight_map entries");

  SafetensorsIndex idx;
  std::set<std::string> distinct;
  for (const auto& [tensor, shard] : wm->members()) {
    if (tensor.empty() || tensor.size() > limits.max_name_bytes) {
      return fail("safetensors index: bad tensor name length");
    }
    if (!shard.is_string()) {
      return fail("safetensors index: weight_map value for \"" + tensor + "\" is not a string");
    }
    const std::string& file = shard.text();
    if (!is_safe_shard_filename(file, limits.max_filename_bytes)) {
      return fail("safetensors index: unsafe shard filename for \"" + tensor + "\"");
    }
    // Duplicate keys are already rejected by the JSON parser, so each insert is new.
    idx.weight_map_.emplace(tensor, file);
    if (distinct.insert(file).second && distinct.size() > limits.max_shards) {
      return fail("safetensors index: too many distinct shard files");
    }
  }
  idx.shards_.assign(distinct.begin(), distinct.end());
  return idx;
}

}  // namespace llmi
