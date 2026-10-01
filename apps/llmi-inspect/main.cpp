// llmi-inspect: load a model directory (config.json + model.safetensors),
// check every weight against the architecture, and print a summary.
//
//   llmi-inspect models/smollm2-135m
//   llmi-inspect models/smollm2-135m --tensor-stats > stats.jsonl
//
// --tensor-stats prints one JSON object per tensor (name, dtype, shape, the
// sum of all values and the first values) so an independent reader can
// confirm the engine reads every byte the same way (tools/crosscheck_weights.py).

#include <cstdio>
#include <exception>
#include <cstring>
#include <map>
#include <string>

#include "llmi/model/model.hpp"
#include "llmi/util/json.hpp"

namespace {

void print_stats(const llmi::Model& m) {
  const auto& w = m.weights();
  for (const auto& t : w.tensors()) {
    const auto raw = w.data(t);
    const auto n = static_cast<std::size_t>(t.numel());
    double sum = 0;
    for (std::size_t i = 0; i < n; ++i) sum += static_cast<double>(llmi::read_float(raw, t.dtype, i));
    std::string line = "{\"name\":";
    llmi::json::append_quoted(line, t.name);
    line += R"(,"dtype":")";
    line += llmi::dtype_name(t.dtype);
    line += R"(","shape":[)";
    for (std::size_t d = 0; d < t.shape.size(); ++d) {
      if (d != 0) line += ',';
      line += std::to_string(t.shape[d]);
    }
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.17g", sum);
    line += "],\"sum\":";
    line += buf;
    line += ",\"first\":[";
    for (std::size_t i = 0; i < std::min<std::size_t>(n, 8); ++i) {
      if (i != 0) line += ',';
      std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(llmi::read_float(raw, t.dtype, i)));
      line += buf;
    }
    line += "]}";
    std::puts(line.c_str());
  }
}

int run(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "--help") == 0) {
    std::fprintf(stderr, "usage: llmi-inspect MODEL_DIR [--tensor-stats]\n");
    return 2;
  }
  const bool stats = argc > 2 && std::strcmp(argv[2], "--tensor-stats") == 0;
  auto model = llmi::Model::load(argv[1]);
  if (!model) {
    std::fprintf(stderr, "error: %s\n", model.error().c_str());
    return 1;
  }
  if (stats) {
    print_stats(model.value());
    return 0;
  }
  const auto& c = model->config();
  std::printf("architecture       %s\n", std::string(llmi::arch_name(c.arch)).c_str());
  std::printf("layers             %u\n", c.num_layers);
  std::printf("hidden size        %u\n", c.hidden_size);
  std::printf("MLP size           %u\n", c.intermediate_size);
  std::printf("attention heads    %u (key/value heads %u, head size %u)\n", c.num_heads, c.num_kv_heads, c.head_dim);
  std::printf("vocabulary         %u\n", c.vocab_size);
  std::printf("context            %u\n", c.max_position_embeddings);
  std::printf("RoPE theta         %g\n", c.rope_theta);
  std::printf("RMSNorm epsilon    %g\n", c.rms_norm_eps);
  std::printf("tied embeddings    %s\n", c.tie_word_embeddings ? "yes" : "no");
  std::printf("q/k/v bias         %s\n", c.qkv_bias ? "yes" : "no");
  std::map<std::string, std::size_t> by_dtype;
  std::uint64_t bytes = 0;
  for (const auto& t : model->weights().tensors()) {
    ++by_dtype[std::string(llmi::dtype_name(t.dtype))];
    bytes += t.end - t.begin;
  }
  std::printf("tensors            %zu (", model->weights().tensors().size());
  bool first = true;
  for (const auto& [k, v] : by_dtype) {
    std::printf("%s%zu %s", first ? "" : ", ", v, k.c_str());
    first = false;
  }
  std::printf(")\n");
  std::printf("parameters         %llu\n", static_cast<unsigned long long>(model->parameter_count()));
  std::printf("weight bytes       %llu\n", static_cast<unsigned long long>(bytes));
  std::printf("all %zu expected tensors present with the expected shapes\n", llmi::expected_tensors(c).size());
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
