// The sharded-safetensors loader (model.safetensors.index.json), checked the
// way this project checks everything: against a reference. The reference here
// is the *same model loaded from a single file* -- a model split across two
// shards must load to bit-identical weights and produce bit-identical logits.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "llmi/model/model.hpp"
#include "llmi/model/transformer.hpp"

using namespace llmi;

namespace {

struct TensorData {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<float> data;
};

// Serializes tensors to the safetensors binary layout (F32): an 8-byte
// little-endian header length, that many bytes of JSON, then the data section,
// tensors laid out contiguously in the given order.
std::string serialize(const std::vector<const TensorData*>& tensors) {
  std::string header = "{";
  std::uint64_t off = 0;
  std::string data;
  for (std::size_t i = 0; i < tensors.size(); ++i) {
    const TensorData& t = *tensors[i];
    std::string shape;
    for (std::size_t d = 0; d < t.shape.size(); ++d) {
      shape += (d != 0 ? "," : "") + std::to_string(t.shape[d]);
    }
    const std::uint64_t nbytes = static_cast<std::uint64_t>(t.data.size()) * sizeof(float);
    header += (i != 0 ? "," : "") + ("\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[" + shape +
                                     "],\"data_offsets\":[" + std::to_string(off) + "," +
                                     std::to_string(off + nbytes) + "]}");
    off += nbytes;
    const auto base = data.size();
    data.resize(base + t.data.size() * sizeof(float));
    std::memcpy(data.data() + base, t.data.data(), t.data.size() * sizeof(float));
  }
  header += "}";

  std::string out;
  out.resize(8);
  const auto hlen = static_cast<std::uint64_t>(header.size());
  for (std::size_t k = 0; k < 8; ++k) out[k] = static_cast<char>((hlen >> (8U * k)) & 0xFFU);
  out += header;
  out += data;
  return out;
}

void write_file(const std::filesystem::path& path, const std::string& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

const char* kTinyConfig = R"({
  "architectures": ["LlamaForCausalLM"], "vocab_size": 10, "hidden_size": 8, "intermediate_size": 12,
  "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 64,
  "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": true, "hidden_act": "silu"
})";

// Deterministic weights for every tensor the tiny config expects.
std::vector<TensorData> make_weights(const ModelConfig& cfg) {
  std::vector<TensorData> out;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> u(-0.1F, 0.1F);
  for (const auto& e : expected_tensors(cfg)) {
    TensorData t;
    t.name = e.name;
    t.shape = e.shape;
    std::uint64_t n = 1;
    for (const std::uint64_t d : e.shape) n *= d;
    t.data.resize(n);
    for (auto& v : t.data) v = u(rng);
    out.push_back(std::move(t));
  }
  return out;
}

std::filesystem::path fresh_dir(const std::string& name) {
  std::filesystem::path dir = std::filesystem::temp_directory_path() / ("llmi-shard-" + name);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

}  // namespace

TEST(ShardedModel, LoadsIdenticallyToASingleFile) {
  auto cfg = parse_config(kTinyConfig);
  ASSERT_TRUE(cfg) << cfg.error();
  const auto weights = make_weights(*cfg);

  // Single-file model.
  const auto single_dir = fresh_dir("single");
  write_file(single_dir / "config.json", kTinyConfig);
  {
    std::vector<const TensorData*> all;
    for (const auto& t : weights) all.push_back(&t);
    write_file(single_dir / "model.safetensors", serialize(all));
  }

  // The same weights split across two shards: the embedding table in shard 1,
  // everything else in shard 2.
  const auto sharded_dir = fresh_dir("sharded");
  write_file(sharded_dir / "config.json", kTinyConfig);
  const std::string shard1 = "model-00001-of-00002.safetensors";
  const std::string shard2 = "model-00002-of-00002.safetensors";
  {
    std::vector<const TensorData*> g1, g2;
    std::string index = R"({"metadata":{"format":"pt"},"weight_map":{)";
    bool first = true;
    for (const auto& t : weights) {
      const bool in1 = t.name == "model.embed_tokens.weight";
      (in1 ? g1 : g2).push_back(&t);
      index += (first ? "" : ",");
      index += "\"" + t.name + "\":\"" + (in1 ? shard1 : shard2) + "\"";
      first = false;
    }
    index += "}}";
    write_file(sharded_dir / shard1, serialize(g1));
    write_file(sharded_dir / shard2, serialize(g2));
    write_file(sharded_dir / "model.safetensors.index.json", index);
  }

  auto single = Model::load(single_dir.string());
  ASSERT_TRUE(single) << single.error();
  auto sharded = Model::load(sharded_dir.string());
  ASSERT_TRUE(sharded) << sharded.error();

  EXPECT_EQ(single->weights().shard_count(), 1U);
  EXPECT_EQ(sharded->weights().shard_count(), 2U);
  EXPECT_EQ(single->parameter_count(), sharded->parameter_count());

  // Every tensor's bytes are identical, whichever file they came from.
  for (const auto& t : weights) {
    const TensorInfo* a = single->weights().find(t.name);
    const TensorInfo* b = sharded->weights().find(t.name);
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr) << t.name;
    const auto da = single->weights().data(*a);
    const auto db = sharded->weights().data(*b);
    ASSERT_EQ(da.size(), db.size()) << t.name;
    EXPECT_EQ(std::memcmp(da.data(), db.data(), da.size()), 0) << t.name;
  }

  // And the forward pass agrees bit for bit.
  auto ta = Transformer::load(*single);
  ASSERT_TRUE(ta) << ta.error();
  auto tb = Transformer::load(*sharded);
  ASSERT_TRUE(tb) << tb.error();
  const std::vector<TokenId> tokens = {1, 5, 2, 7, 0, 3};
  auto la = ta->forward(tokens, /*all_positions=*/true);
  auto lb = tb->forward(tokens, /*all_positions=*/true);
  ASSERT_TRUE(la) << la.error();
  ASSERT_TRUE(lb) << lb.error();
  EXPECT_EQ(la.value(), lb.value());
}

TEST(ShardedModel, RejectsIndexAndShardsThatDisagree) {
  auto cfg = parse_config(kTinyConfig);
  ASSERT_TRUE(cfg);
  const auto weights = make_weights(*cfg);
  const std::string shard1 = "model-00001-of-00002.safetensors";
  const std::string shard2 = "model-00002-of-00002.safetensors";

  // The honest per-tensor map: the embedding table in shard 1, the rest in 2.
  const auto index_of = [&](const TensorData& t) {
    return t.name == "model.embed_tokens.weight" ? shard1 : shard2;
  };
  auto write_index = [&](const std::filesystem::path& dir) {
    std::string index = R"({"weight_map":{)";
    bool first = true;
    for (const auto& t : weights) {
      index += (first ? "" : ",");
      index += "\"" + t.name + "\":\"" + index_of(t) + "\"";
      first = false;
    }
    index += "}}";
    write_file(dir / "config.json", kTinyConfig);
    write_file(dir / "model.safetensors.index.json", index);
  };

  // Case 1: the index promises a tensor is in a shard that does not hold it.
  // Here shard 2 is written empty, so every tensor the map sends to shard 2
  // is missing from it.
  {
    const auto dir = fresh_dir("not-there");
    write_index(dir);
    std::vector<const TensorData*> g1;
    for (const auto& t : weights) {
      if (index_of(t) == shard1) g1.push_back(&t);
    }
    write_file(dir / shard1, serialize(g1));
    write_file(dir / shard2, serialize({}));  // empty: the promised tensors are absent
    auto m = Model::load(dir.string());
    ASSERT_FALSE(m);
    EXPECT_NE(m.error().find("is not there"), std::string::npos) << m.error();
  }

  // Case 2: every map entry is satisfiable, but a shard physically holds a
  // tensor the index attributes to the *other* shard. Shard 1 carries an
  // extra copy of a layer-0 weight that the map assigns to shard 2 (where it
  // also lives, so the first check passes).
  {
    const auto dir = fresh_dir("unlisted");
    write_index(dir);
    const TensorData* stowaway = nullptr;
    std::vector<const TensorData*> g1, g2;
    for (const auto& t : weights) {
      if (index_of(t) == shard1) {
        g1.push_back(&t);
      } else {
        g2.push_back(&t);
        if (t.name == "model.layers.0.input_layernorm.weight") stowaway = &t;
      }
    }
    ASSERT_NE(stowaway, nullptr);
    g1.push_back(stowaway);  // an extra copy in shard 1, which the index does not list there
    write_file(dir / shard1, serialize(g1));
    write_file(dir / shard2, serialize(g2));
    auto m = Model::load(dir.string());
    ASSERT_FALSE(m);
    EXPECT_NE(m.error().find("does not list"), std::string::npos) << m.error();
  }
}

TEST(ShardedModel, RejectsAMissingShardFileAndAnEmptyDir) {
  auto cfg = parse_config(kTinyConfig);
  ASSERT_TRUE(cfg);
  const auto weights = make_weights(*cfg);

  // An index naming a shard file that does not exist.
  const auto dir = fresh_dir("missing-shard");
  write_file(dir / "config.json", kTinyConfig);
  std::string index = R"({"weight_map":{)";
  bool first = true;
  for (const auto& t : weights) {
    index += (first ? "" : ",");
    index += "\"" + t.name + "\":\"model-00001-of-00001.safetensors\"";
    first = false;
  }
  index += "}}";
  write_file(dir / "model.safetensors.index.json", index);  // but no shard file written
  EXPECT_FALSE(Model::load(dir.string()));

  // A directory with neither a single file nor an index.
  const auto empty = fresh_dir("empty");
  write_file(empty / "config.json", kTinyConfig);
  auto m = Model::load(empty.string());
  ASSERT_FALSE(m);
  EXPECT_NE(m.error().find("neither"), std::string::npos) << m.error();
}
