#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "llmi/model/safetensors.hpp"
#include "llmi/util/result.hpp"

namespace llmi {

// The architectures this engine knows. Both are decoder-only transformers
// with RMSNorm, rotary position embeddings, grouped-query attention and a
// SwiGLU MLP; Qwen2 adds a bias to the query, key and value projections.
enum class Arch : std::uint8_t { Llama, Qwen2 };

std::string_view arch_name(Arch a);

// Hyperparameters from a Hugging Face config.json, validated.
struct ModelConfig {
  Arch arch = Arch::Llama;
  std::uint32_t vocab_size = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t intermediate_size = 0;
  std::uint32_t num_layers = 0;
  std::uint32_t num_heads = 0;
  std::uint32_t num_kv_heads = 0;
  std::uint32_t head_dim = 0;
  std::uint32_t max_position_embeddings = 0;
  double rms_norm_eps = 0;
  double rope_theta = 0;
  bool tie_word_embeddings = false;
  bool qkv_bias = false;
  std::vector<std::int32_t> eos_token_ids;  // from eos_token_id (a number or a list); may be empty
};

// Parses and validates config.json text. Unsupported features (other
// architectures, RoPE scaling, sliding-window attention) are reported as
// errors rather than silently ignored.
Result<ModelConfig> parse_config(std::string_view json_text);

// One weight the architecture requires, with its expected shape.
struct ExpectedTensor {
  std::string name;
  std::vector<std::uint64_t> shape;
};

// Every tensor the forward pass will read, in a fixed order.
std::vector<ExpectedTensor> expected_tensors(const ModelConfig& cfg);

// A model directory: config.json + model.safetensors, checked against each
// other: every expected tensor present with the right shape and a float
// dtype, and no unexpected tensors.
class Model {
 public:
  static Result<Model> load(const std::string& dir);
  // For tests and fuzzing: check an already-parsed file against a config.
  static Result<Model> from_parts(ModelConfig cfg, SafetensorsFile weights);

  [[nodiscard]] const ModelConfig& config() const { return cfg_; }
  [[nodiscard]] const SafetensorsFile& weights() const { return weights_; }
  [[nodiscard]] std::uint64_t parameter_count() const;

 private:
  ModelConfig cfg_;
  SafetensorsFile weights_;
};

Result<std::string> read_file(const std::string& path, std::size_t max_bytes);

}  // namespace llmi
