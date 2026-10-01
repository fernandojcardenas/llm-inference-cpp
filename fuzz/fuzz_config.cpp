// Arbitrary bytes as config.json and as tokenizer.json. Invariants: no crash
// or sanitizer report; an accepted config is internally consistent.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "llmi/model/model.hpp"
#include "llmi/tokenizer/tokenizer.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  if (auto c = llmi::parse_config(text)) {
    if (c->head_dim * c->num_heads != c->hidden_size || c->num_heads % c->num_kv_heads != 0) std::abort();
    if (llmi::expected_tensors(c.value()).size() < 3) std::abort();
  }
  (void)llmi::Tokenizer::from_json(text);
  return 0;
}
