#include "llmi/model/model.hpp"

#include <gtest/gtest.h>

#include <cstring>

using namespace llmi;

namespace {

const char* kTinyConfig = R"({
  "architectures": ["LlamaForCausalLM"], "vocab_size": 10, "hidden_size": 8, "intermediate_size": 12,
  "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2, "max_position_embeddings": 64,
  "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "tie_word_embeddings": true, "hidden_act": "silu",
  "rope_scaling": null
})";

// Builds an in-memory safetensors file holding exactly `tensors` (BF16 zeros).
struct FileBuilder {
  std::vector<std::byte> bytes;
  Result<SafetensorsFile> build(const std::vector<ExpectedTensor>& tensors, DType dtype = DType::BF16) {
    std::string header = "{";
    std::uint64_t off = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
      std::uint64_t n = dtype_size(dtype);
      std::string shape;
      for (std::size_t d = 0; d < tensors[i].shape.size(); ++d) {
        n *= tensors[i].shape[d];
        shape += (d != 0 ? "," : "") + std::to_string(tensors[i].shape[d]);
      }
      header += (i != 0 ? "," : "") + std::string("\"") + tensors[i].name + "\":{\"dtype\":\"" +
                std::string(dtype_name(dtype)) + "\",\"shape\":[" + shape + "],\"data_offsets\":[" +
                std::to_string(off) + "," + std::to_string(off + n) + "]}";
      off += n;
    }
    header += "}";
    bytes.assign(8 + header.size() + off, std::byte{0});
    for (int k = 0; k < 8; ++k) bytes[static_cast<std::size_t>(k)] = std::byte((header.size() >> (8U * static_cast<unsigned>(k))) & 0xFFU);
    std::memcpy(bytes.data() + 8, header.data(), header.size());
    return SafetensorsFile::parse(bytes);
  }
};

}  // namespace

TEST(Config, ParsesALlamaConfig) {
  auto c = parse_config(kTinyConfig);
  ASSERT_TRUE(c) << c.error();
  EXPECT_EQ(c->arch, Arch::Llama);
  EXPECT_EQ(c->head_dim, 2U);
  EXPECT_EQ(c->num_kv_heads, 2U);
  EXPECT_TRUE(c->tie_word_embeddings);
  EXPECT_FALSE(c->qkv_bias);
}

TEST(Config, ParsesTheRealSmolLM2Config) {
  auto text = read_file(std::string(LLMI_TESTDATA_DIR) + "/smollm2-135m/config.json", 1U << 20U);
  ASSERT_TRUE(text) << text.error();
  auto c = parse_config(text.value());
  ASSERT_TRUE(c) << c.error();
  EXPECT_EQ(c->num_layers, 30U);
  EXPECT_EQ(c->hidden_size, 576U);
  EXPECT_EQ(c->num_heads, 9U);
  EXPECT_EQ(c->num_kv_heads, 3U);
  EXPECT_EQ(c->head_dim, 64U);
  EXPECT_EQ(c->vocab_size, 49152U);
  EXPECT_EQ(c->rope_theta, 100000.0);
}

TEST(Config, Qwen2HasQkvBias) {
  std::string cfg = kTinyConfig;
  cfg.replace(cfg.find("LlamaForCausalLM"), 16, "Qwen2ForCausalLM");
  auto c = parse_config(cfg);
  ASSERT_TRUE(c) << c.error();
  EXPECT_TRUE(c->qkv_bias);
  EXPECT_EQ(expected_tensors(*c).size(), 1U + 2U * 12U + 1U);
}

TEST(Config, RejectsUnsupportedOrInconsistentConfigs) {
  auto with = [](const std::string& from, const std::string& to) {
    std::string cfg = kTinyConfig;
    cfg.replace(cfg.find(from), from.size(), to);
    return parse_config(cfg);
  };
  EXPECT_FALSE(with("LlamaForCausalLM", "MistralForCausalLM"));
  EXPECT_FALSE(with("\"hidden_size\": 8", "\"hidden_size\": 9"));             // not divisible by heads
  EXPECT_FALSE(with("\"num_key_value_heads\": 2", "\"num_key_value_heads\": 3"));
  EXPECT_FALSE(with("\"hidden_size\": 8", "\"hidden_size\": 0"));
  EXPECT_FALSE(with("\"hidden_size\": 8", "\"hidden_size\": 8.5"));
  EXPECT_FALSE(with("\"hidden_size\": 8", "\"hidden_size\": 99999999999"));
  EXPECT_FALSE(with("\"rope_scaling\": null", "\"rope_scaling\": {\"type\": \"linear\"}"));
  EXPECT_FALSE(with("\"silu\"", "\"gelu\""));
  EXPECT_FALSE(with("\"rms_norm_eps\": 1e-5", "\"rms_norm_eps\": -1"));
  EXPECT_FALSE(with("\"vocab_size\": 10, ", ""));                              // missing
  EXPECT_FALSE(parse_config("[]"));
}

TEST(Model, AcceptsMatchingWeights) {
  auto c = parse_config(kTinyConfig);
  ASSERT_TRUE(c);
  FileBuilder fb;
  auto st = fb.build(expected_tensors(*c));
  ASSERT_TRUE(st) << st.error();
  auto m = Model::from_parts(*c, std::move(st.value()));
  ASSERT_TRUE(m) << m.error();
  // embed 10x8 + per layer (8 + 8x8 + 4x8 + 4x8 + 8x8 + 8 + 3 x 12x8) + norm 8
  EXPECT_EQ(m->parameter_count(), 80U + 2U * (8U + 64U + 32U + 32U + 64U + 8U + 288U) + 8U);
}

TEST(Model, RejectsMissingWrongOrExtraWeights) {
  auto c = parse_config(kTinyConfig);
  ASSERT_TRUE(c);
  const auto good = expected_tensors(*c);

  auto missing = good;
  missing.erase(missing.begin() + 3);
  FileBuilder f1;
  auto r1 = Model::from_parts(*c, std::move(f1.build(missing).value()));
  ASSERT_FALSE(r1);
  EXPECT_NE(r1.error().find("missing tensor"), std::string::npos);

  auto wrong = good;
  wrong[2].shape = {4, 9};
  FileBuilder f2;
  auto r2 = Model::from_parts(*c, std::move(f2.build(wrong).value()));
  ASSERT_FALSE(r2);
  EXPECT_NE(r2.error().find("has shape"), std::string::npos);

  auto extra = good;
  extra.push_back({"model.layers.9.mlp.up_proj.weight", {12, 8}});
  FileBuilder f3;
  EXPECT_FALSE(Model::from_parts(*c, std::move(f3.build(extra).value())));

  FileBuilder f4;
  auto r4 = Model::from_parts(*c, std::move(f4.build(good, DType::I8).value()));
  ASSERT_FALSE(r4);
  EXPECT_NE(r4.error().find("dtype"), std::string::npos);
}
