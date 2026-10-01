// Arbitrary bytes as a safetensors file. Invariants: no crash or sanitizer
// report; in an accepted file every tensor's bytes lie inside the input,
// match dtype x shape, and the tensors tile the data section exactly.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "llmi/model/safetensors.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(data), size);
  auto f = llmi::SafetensorsFile::parse(bytes);
  if (!f) return 0;
  std::uint64_t total = 0;
  for (const auto& t : f->tensors()) {
    const auto raw = f->data(t);
    if (raw.data() < bytes.data() || raw.data() + raw.size() > bytes.data() + bytes.size()) std::abort();
    if (raw.size() != t.numel() * llmi::dtype_size(t.dtype)) std::abort();
    if (f->find(t.name) != &t) std::abort();
    if (!raw.empty() && (t.dtype == llmi::DType::F32 || t.dtype == llmi::DType::F16 || t.dtype == llmi::DType::BF16)) {
      (void)llmi::read_float(raw, t.dtype, t.numel() - 1);  // last element must be readable
    }
    total += raw.size();
  }
  std::uint64_t header = 0;
  for (int k = 7; k >= 0; --k) header = (header << 8U) | data[k];
  if (8 + header + total != size) std::abort();
  return 0;
}
