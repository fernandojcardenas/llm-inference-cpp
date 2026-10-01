// Arbitrary bytes into the JSON parser. Invariants: no crash, no sanitizer
// report, and every string in an accepted document is valid UTF-8.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "llmi/util/json.hpp"
#include "llmi/util/utf8.hpp"

namespace {
void check(const llmi::json::Value& v) {
  if (v.is_string() && !llmi::utf8::valid(v.text())) std::abort();
  for (const auto& item : v.items()) check(item);
  for (const auto& [k, m] : v.members()) {
    if (!llmi::utf8::valid(k)) std::abort();
    check(m);
  }
  if (v.is_number()) {
    (void)v.as_u64();
    (void)v.as_i64();
    (void)v.as_double();
  }
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  auto v = llmi::json::parse(std::string_view(reinterpret_cast<const char*>(data), size));
  if (v) check(v.value());
  return 0;
}
