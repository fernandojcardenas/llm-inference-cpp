// llmi-gguf-inspect: open a GGUF v3 file with the hardened reader
// (src/model/gguf.cpp, M6) and print a summary, or one JSON line per tensor
// for an independent reader to cross-check (tools/crosscheck_gguf.py).
//
//   llmi-gguf-inspect model.gguf
//   llmi-gguf-inspect model.gguf --tensor-stats > stats.jsonl
//
// This tool does not feed a GGUF file into Transformer -- M6's scope is the
// loader itself (reading GGUF safely), not switching inference over to it;
// see docs/gguf.md.

#include <cstdio>
#include <cstring>
#include <exception>
#include <string>

#include "llmi/model/gguf.hpp"
#include "llmi/util/json.hpp"

namespace {

// Returns false (after printing an error to stderr) if any tensor's
// conversion fails, so the caller can return a normal error code instead of
// calling std::exit from inside a loop over user-supplied data.
bool print_stats(const llmi::gguf::GGUFFile& f) {
  for (const auto& t : f.tensors()) {
    std::string line = "{\"name\":";
    llmi::json::append_quoted(line, t.name);
    line += R"(,"type":")";
    line += llmi::gguf::tensor_type_name(t.type);
    line += R"(","shape":[)";
    for (std::size_t d = 0; d < t.shape.size(); ++d) {
      if (d != 0) line += ',';
      line += std::to_string(t.shape[d]);
    }
    line += "],\"n_elements\":" + std::to_string(t.numel());
    line += ",\"data_nbytes\":" + std::to_string(t.size);
    line += ",\"offset\":" + std::to_string(t.offset);

    char buf[64];
    if (t.type == llmi::gguf::TensorType::F32 || t.type == llmi::gguf::TensorType::F16 ||
        t.type == llmi::gguf::TensorType::BF16) {
      auto vals = llmi::gguf::to_f32(f, t);
      if (!vals) {
        std::fprintf(stderr, "error: %s\n", vals.error().c_str());
        return false;
      }
      double sum = 0;
      for (const float v : *vals) sum += static_cast<double>(v);
      std::snprintf(buf, sizeof buf, "%.17g", sum);
      line += ",\"sum\":";
      line += buf;
      line += ",\"first\":[";
      for (std::size_t i = 0; i < std::min<std::size_t>(vals->size(), 8); ++i) {
        if (i != 0) line += ',';
        std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>((*vals)[i]));
        line += buf;
      }
      line += "]";
    } else if (t.type == llmi::gguf::TensorType::Q8_0 || t.type == llmi::gguf::TensorType::Q4_0) {
      // Flattened as one giant row: block boundaries are every 32 elements
      // of the tensor's flat storage order regardless of its logical shape
      // (docs/gguf.md), so out=1/in=numel reads every block exactly once.
      auto m = llmi::gguf::to_quantized(f, t, 1, t.numel());
      if (!m) {
        std::fprintf(stderr, "error: %s\n", m.error().c_str());
        return false;
      }
      std::vector<float> row(t.numel());
      llmi::quant::dequantize_row(*m, 0, row.data());
      double sum = 0;
      for (const float v : row) sum += static_cast<double>(v);
      std::snprintf(buf, sizeof buf, "%.17g", sum);
      line += ",\"sum\":";
      line += buf;
      line += ",\"first\":[";
      for (std::size_t i = 0; i < std::min<std::size_t>(row.size(), 8); ++i) {
        if (i != 0) line += ',';
        std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(row[i]));
        line += buf;
      }
      line += "]";
    }
    line += "}";
    std::puts(line.c_str());
  }
  return true;
}

int run(int argc, char** argv) {
  if (argc < 2 || std::strcmp(argv[1], "--help") == 0) {
    std::fprintf(stderr, "usage: llmi-gguf-inspect FILE.gguf [--tensor-stats]\n");
    return 2;
  }
  const bool stats = argc > 2 && std::strcmp(argv[2], "--tensor-stats") == 0;
  auto f = llmi::gguf::GGUFFile::open(argv[1]);
  if (!f) {
    std::fprintf(stderr, "error: %s\n", f.error().c_str());
    return 1;
  }
  if (stats) {
    return print_stats(*f) ? 0 : 1;
  }
  std::printf("version            %u\n", f->version());
  std::printf("alignment          %llu\n", static_cast<unsigned long long>(f->alignment()));
  std::printf("metadata entries   %zu\n", f->metadata().size());
  std::printf("tensors            %zu\n", f->tensors().size());
  if (const auto* arch = f->find_metadata("general.architecture")) {
    std::printf("architecture       %s\n", arch->as_string.c_str());
  }
  if (const auto* name = f->find_metadata("general.name")) {
    std::printf("name               %s\n", name->as_string.c_str());
  }
  std::uint64_t data_bytes = 0;
  for (const auto& t : f->tensors()) data_bytes += t.size;
  std::printf("tensor data bytes  %llu\n", static_cast<unsigned long long>(data_bytes));
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
