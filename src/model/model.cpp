#include "llmi/model/model.hpp"

#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#include "llmi/model/safetensors_index.hpp"
#include "llmi/model/weights.hpp"
#include "llmi/util/json.hpp"

namespace llmi {

namespace {

bool file_exists(const std::string& path) {
  const std::ifstream in(path, std::ios::binary);
  return in.good();
}

}  // namespace

std::string_view arch_name(Arch a) {
  switch (a) {
    case Arch::Llama: return "LlamaForCausalLM";
    case Arch::Qwen2: return "Qwen2ForCausalLM";
  }
  return "?";
}

namespace {

// Upper bounds for any field; far above the models this engine targets,
// low enough that products of them cannot overflow 64 bits.
constexpr std::uint64_t kMaxDim = 1U << 20U;
constexpr std::uint64_t kMaxVocab = 1U << 22U;
constexpr std::uint64_t kMaxLayers = 1024;
constexpr std::uint64_t kMaxPositions = 1U << 24U;

Result<std::uint32_t> get_uint(const json::Value& root, std::string_view key, std::uint64_t max,
                               std::optional<std::uint32_t> fallback = std::nullopt) {
  const json::Value* v = root.find(key);
  if (v == nullptr || v->is_null()) {
    if (fallback) return *fallback;
    return fail("config: missing \"" + std::string(key) + "\"");
  }
  auto n = v->as_u64();
  if (!n || *n == 0 || *n > max) return fail("config: \"" + std::string(key) + "\" must be an integer in 1.." + std::to_string(max));
  return static_cast<std::uint32_t>(*n);
}

Result<double> get_positive(const json::Value& root, std::string_view key) {
  const json::Value* v = root.find(key);
  if (v == nullptr) return fail("config: missing \"" + std::string(key) + "\"");
  auto d = v->as_double();
  if (!d || *d <= 0) return fail("config: \"" + std::string(key) + "\" must be a positive number");
  return *d;
}

Result<bool> get_bool(const json::Value& root, std::string_view key, bool fallback) {
  const json::Value* v = root.find(key);
  if (v == nullptr || v->is_null()) return fallback;
  if (!v->is_bool()) return fail("config: \"" + std::string(key) + "\" must be true or false");
  return v->as_bool();
}

}  // namespace

Result<ModelConfig> parse_config(std::string_view json_text) {
  auto root = json::parse(json_text);
  if (!root) return fail("config: " + root.error());
  if (!root->is_object()) return fail("config: not a JSON object");

  ModelConfig c;
  const json::Value* archs = root->find("architectures");
  if (archs == nullptr || !archs->is_array() || archs->items().size() != 1 || !archs->items()[0].is_string()) {
    return fail("config: \"architectures\" must list exactly one architecture");
  }
  const std::string& a = archs->items()[0].text();
  if (a == "LlamaForCausalLM") {
    c.arch = Arch::Llama;
  } else if (a == "Qwen2ForCausalLM") {
    c.arch = Arch::Qwen2;
  } else {
    return fail("config: unsupported architecture " + a);
  }

#define LLMI_TRY(var, expr)                    \
  auto var##_r = (expr);                       \
  if (!var##_r) return fail(var##_r.error());  \
  c.var = var##_r.value()

  LLMI_TRY(vocab_size, get_uint(*root, "vocab_size", kMaxVocab));
  LLMI_TRY(hidden_size, get_uint(*root, "hidden_size", kMaxDim));
  LLMI_TRY(intermediate_size, get_uint(*root, "intermediate_size", kMaxDim));
  LLMI_TRY(num_layers, get_uint(*root, "num_hidden_layers", kMaxLayers));
  LLMI_TRY(num_heads, get_uint(*root, "num_attention_heads", kMaxDim));
  LLMI_TRY(num_kv_heads, get_uint(*root, "num_key_value_heads", kMaxDim, c.num_heads));
  LLMI_TRY(max_position_embeddings, get_uint(*root, "max_position_embeddings", kMaxPositions));
  LLMI_TRY(rms_norm_eps, get_positive(*root, "rms_norm_eps"));
  LLMI_TRY(rope_theta, get_positive(*root, "rope_theta"));
  LLMI_TRY(tie_word_embeddings, get_bool(*root, "tie_word_embeddings", false));
  if (c.arch == Arch::Llama) {
    LLMI_TRY(qkv_bias, get_bool(*root, "attention_bias", false));
  } else {
    c.qkv_bias = true;  // Qwen2 always has q/k/v biases
  }
#undef LLMI_TRY

  if (c.hidden_size % c.num_heads != 0) return fail("config: hidden_size is not a multiple of num_attention_heads");
  c.head_dim = c.hidden_size / c.num_heads;
  if (const json::Value* hd = root->find("head_dim"); hd != nullptr && !hd->is_null()) {
    auto v = hd->as_u64();
    if (!v || *v != c.head_dim) return fail("config: head_dim other than hidden_size / num_attention_heads is not supported");
  }
  if (c.head_dim % 2 != 0) return fail("config: head_dim must be even for rotary embeddings");
  if (c.num_heads % c.num_kv_heads != 0) return fail("config: num_attention_heads is not a multiple of num_key_value_heads");

  if (const json::Value* act = root->find("hidden_act"); act != nullptr && !(act->is_string() && act->text() == "silu")) {
    return fail("config: only hidden_act \"silu\" is supported");
  }
  if (const json::Value* eos = root->find("eos_token_id"); eos != nullptr && !eos->is_null()) {
    std::vector<const json::Value*> ids;
    if (eos->is_array()) {
      for (const auto& e : eos->items()) ids.push_back(&e);
    } else {
      ids.push_back(eos);
    }
    for (const auto* e : ids) {
      auto n = e->as_u64();
      if (!n || *n >= c.vocab_size) return fail("config: eos_token_id out of range");
      c.eos_token_ids.push_back(static_cast<std::int32_t>(*n));
    }
  }
  if (const json::Value* rs = root->find("rope_scaling"); rs != nullptr && !rs->is_null()) {
    return fail("config: rope_scaling is not supported");
  }
  if (const json::Value* sw = root->find("use_sliding_window"); sw != nullptr && sw->is_bool() && sw->as_bool()) {
    return fail("config: sliding-window attention is not supported");
  }
  return c;
}

std::vector<ExpectedTensor> expected_tensors(const ModelConfig& c) {
  const std::uint64_t h = c.hidden_size;
  const std::uint64_t q = static_cast<std::uint64_t>(c.num_heads) * c.head_dim;
  const std::uint64_t kv = static_cast<std::uint64_t>(c.num_kv_heads) * c.head_dim;
  const std::uint64_t ff = c.intermediate_size;
  std::vector<ExpectedTensor> out;
  out.push_back({"model.embed_tokens.weight", {c.vocab_size, h}});
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";
    out.push_back({p + "input_layernorm.weight", {h}});
    out.push_back({p + "self_attn.q_proj.weight", {q, h}});
    out.push_back({p + "self_attn.k_proj.weight", {kv, h}});
    out.push_back({p + "self_attn.v_proj.weight", {kv, h}});
    if (c.qkv_bias) {
      out.push_back({p + "self_attn.q_proj.bias", {q}});
      out.push_back({p + "self_attn.k_proj.bias", {kv}});
      out.push_back({p + "self_attn.v_proj.bias", {kv}});
    }
    out.push_back({p + "self_attn.o_proj.weight", {h, q}});
    out.push_back({p + "post_attention_layernorm.weight", {h}});
    out.push_back({p + "mlp.gate_proj.weight", {ff, h}});
    out.push_back({p + "mlp.up_proj.weight", {ff, h}});
    out.push_back({p + "mlp.down_proj.weight", {h, ff}});
  }
  out.push_back({"model.norm.weight", {h}});
  if (!c.tie_word_embeddings) out.push_back({"lm_head.weight", {c.vocab_size, h}});
  return out;
}

namespace {

std::string shape_str(const std::vector<std::uint64_t>& s) {
  std::string out = "[";
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (i != 0) out += ", ";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

}  // namespace

Result<Model> Model::from_parts(ModelConfig cfg, SafetensorsFile weights) {
  return from_parts(std::move(cfg), Weights::single(std::move(weights)));
}

Result<Model> Model::from_parts(ModelConfig cfg, Weights weights) {
  std::set<std::string> expected_names;
  for (const auto& e : expected_tensors(cfg)) {
    const TensorInfo* t = weights.find(e.name);
    if (t == nullptr) return fail("model: missing tensor " + e.name);
    if (t->shape != e.shape) {
      return fail("model: " + e.name + " has shape " + shape_str(t->shape) + ", expected " + shape_str(e.shape));
    }
    if (t->dtype != DType::F32 && t->dtype != DType::F16 && t->dtype != DType::BF16) {
      return fail("model: " + e.name + " has dtype " + std::string(dtype_name(t->dtype)) + ", expected F32, F16 or BF16");
    }
    expected_names.insert(e.name);
  }
  for (const auto& t : weights.tensors()) {
    // Some exports also store the tied output matrix or rotary caches; anything
    // else means the file is not the model the config describes.
    if (!expected_names.contains(t.name) && t.name != "lm_head.weight" &&
        t.name.find("rotary_emb.inv_freq") == std::string::npos) {
      return fail("model: unexpected tensor " + t.name);
    }
  }
  Model m;
  m.cfg_ = std::move(cfg);
  m.weights_ = std::move(weights);
  return m;
}

Result<std::string> read_file(const std::string& path, std::size_t max_bytes) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return fail("cannot open " + path);
  std::string data;
  char buf[65536];
  while (in.read(buf, sizeof buf) || in.gcount() > 0) {
    data.append(buf, static_cast<std::size_t>(in.gcount()));
    if (data.size() > max_bytes) return fail(path + " is larger than " + std::to_string(max_bytes) + " bytes");
  }
  return data;
}

// Reads model.safetensors.index.json, opens every shard it names, and checks
// that the index and the shards describe exactly the same tensors: each
// weight_map entry is present in the shard it points to, and every tensor in
// every shard is attributed to that same shard by the index. Either mismatch
// means the download is inconsistent, so it is an error, not a silent merge.
Result<Weights> load_sharded(const std::string& dir) {
  auto text = read_file(dir + "/model.safetensors.index.json", 64U << 20U);
  if (!text) return fail(text.error());
  auto index = SafetensorsIndex::parse(text.value());
  if (!index) return fail(dir + "/model.safetensors.index.json: " + index.error());

  std::vector<SafetensorsFile> shards;
  std::map<std::string, std::size_t> shard_pos;  // filename -> index into shards
  for (const std::string& name : index->shards()) {
    std::string path = dir + "/";
    path += name;
    auto f = SafetensorsFile::open(path);
    if (!f) return fail(f.error());
    shard_pos.emplace(name, shards.size());
    shards.push_back(std::move(f.value()));
  }

  // Every tensor the index promises is in the shard the index names.
  for (const auto& [tensor, file] : index->weight_map()) {
    if (shards[shard_pos.at(file)].find(tensor) == nullptr) {
      std::string msg = "model: index lists tensor \"";
      msg.append(tensor).append("\" in ").append(file).append(", but it is not there");
      return fail(msg);
    }
  }
  // Every tensor a shard actually holds is attributed to that shard by the index.
  for (const std::string& name : index->shards()) {
    const SafetensorsFile& shard = shards[shard_pos.at(name)];
    for (const auto& t : shard.tensors()) {
      auto it = index->weight_map().find(t.name);
      if (it == index->weight_map().end() || it->second != name) {
        std::string msg = "model: shard ";
        msg.append(name).append(" holds tensor \"").append(t.name).append("\", which the index does not list there");
        return fail(msg);
      }
    }
  }
  return Weights::sharded(std::move(shards));
}

Result<Model> Model::load(const std::string& dir) {
  auto text = read_file(dir + "/config.json", 1U << 20U);
  if (!text) return fail(text.error());
  auto cfg = parse_config(text.value());
  if (!cfg) return fail(dir + "/config.json: " + cfg.error());

  // A single-file model takes precedence; a corrupt model.safetensors is a
  // real error, never a reason to fall back to an index that may not exist.
  if (file_exists(dir + "/model.safetensors")) {
    auto weights = SafetensorsFile::open(dir + "/model.safetensors");
    if (!weights) return fail(weights.error());
    return from_parts(cfg.value(), std::move(weights.value()));
  }
  if (file_exists(dir + "/model.safetensors.index.json")) {
    auto weights = load_sharded(dir);
    if (!weights) return fail(weights.error());
    return from_parts(cfg.value(), std::move(weights.value()));
  }
  return fail("model: " + dir + " has neither model.safetensors nor model.safetensors.index.json");
}

std::uint64_t Model::parameter_count() const {
  std::uint64_t n = 0;
  for (const auto& t : weights_.tensors()) n += t.numel();
  return n;
}

}  // namespace llmi
