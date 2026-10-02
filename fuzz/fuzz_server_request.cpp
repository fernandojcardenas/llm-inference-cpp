// Arbitrary bytes as a /v1/chat/completions JSON request body -- the
// boundary an HTTP client controls directly (ADR 0008 treats it the same as
// any other untrusted input this engine parses). Invariants: no crash or
// sanitizer report, and an accepted request always obeys the limits it was
// parsed against.
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <string_view>

#include "llmi/server/request.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const std::string_view text(reinterpret_cast<const char*>(data), size);
  const llmi::server::RequestLimits limits;  // defaults: the same ceilings a real server enforces
  if (auto req = llmi::server::parse_chat_completion_request(text, limits)) {
    if (req->messages.empty() || req->messages.size() > limits.max_messages) std::abort();
    if (req->max_tokens > limits.max_tokens_ceiling) std::abort();
    if (req->stop.size() > limits.max_stop_sequences) std::abort();
    for (const auto& m : req->messages) {
      if (m.role != "system" && m.role != "user" && m.role != "assistant") std::abort();
      if (m.content.size() > limits.max_message_bytes) std::abort();
    }
  }
  return 0;
}
