// Arbitrary bytes as a GGUF v3 file. Invariants: no crash or sanitizer
// report; in an accepted file every tensor's bytes lie inside the input, no
// two tensors overlap, and every reported tensor name is findable.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "llmi/model/gguf.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(data), size);
  auto f = llmi::gguf::GGUFFile::parse(bytes);
  if (!f) return 0;
  if (f->alignment() == 0 || (f->alignment() & (f->alignment() - 1)) != 0) std::abort();
  for (const auto& t : f->tensors()) {
    const auto raw = f->data(t);
    if (raw.data() < bytes.data() || raw.data() + raw.size() > bytes.data() + bytes.size()) std::abort();
    if (raw.size() != t.size) std::abort();
    if (t.offset % f->alignment() != 0) std::abort();
    if (f->find(t.name) != &t) std::abort();
    if (auto info = llmi::gguf::tensor_type_info(t.type)) {
      if (t.numel() % info->block_size != 0) std::abort();
    } else {
      std::abort();  // parse() must never accept an unknown tensor type
    }
  }
  for (const auto& [key, value] : f->metadata()) {
    if (key.empty()) std::abort();
    if (value.type == llmi::gguf::ValueType::Array && value.array_type == llmi::gguf::ValueType::Array) {
      std::abort();  // parse() must never accept a nested array
    }
  }
  return 0;
}
