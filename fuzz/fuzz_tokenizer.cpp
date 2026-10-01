// Arbitrary bytes into the SmolLM2 tokenizer. Invariants: valid UTF-8 always
// encodes; invalid UTF-8 is rejected; when no byte was dropped, decoding the
// tokens gives back exactly the input; decoding arbitrary ids never crashes.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "llmi/tokenizer/tokenizer.hpp"
#include "llmi/util/utf8.hpp"

namespace {
const llmi::Tokenizer& tok() {
  static const llmi::Tokenizer t = [] {
    auto r = llmi::Tokenizer::load(std::string(LLMI_TESTDATA_DIR) + "/smollm2-135m/tokenizer.json");
    if (!r) std::abort();
    return std::move(r.value());
  }();
  return t;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  if (size == 0) return 0;
  const bool parse_special = (data[0] & 1U) != 0;
  const std::string_view text(reinterpret_cast<const char*>(data + 1), size - 1);
  std::size_t dropped = 0;
  auto ids = tok().encode(text, parse_special, &dropped);
  if (ids.ok() != llmi::utf8::valid(text)) std::abort();
  if (ids && dropped == 0) {
    auto back = tok().decode(ids.value());
    if (!back || back.value() != text) std::abort();
  }
  // The same bytes as a sequence of token ids (some out of range).
  std::vector<llmi::TokenId> raw;
  for (std::size_t i = 1; i + 2 <= size; i += 2) raw.push_back(static_cast<llmi::TokenId>(data[i] | (data[i + 1] << 8U)));
  auto decoded = tok().decode(raw, (data[0] & 2U) != 0);
  bool in_range = true;
  for (const auto id : raw) in_range = in_range && static_cast<std::size_t>(id) < tok().vocab_size();
  if (decoded.ok() != in_range) std::abort();
  return 0;
}
