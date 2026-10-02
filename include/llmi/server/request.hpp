#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "llmi/chat/template.hpp"
#include "llmi/model/sampling.hpp"
#include "llmi/util/result.hpp"

namespace llmi::server {

// Server-wide ceilings, independent of what any one client asks for.
// Enforced regardless of the request's own fields (ADR 0008: generation is
// serialized behind a single slot, so one request's length is every other
// request's wait -- a client cannot opt out of these by asking nicely).
struct RequestLimits {
  std::size_t default_max_tokens = 16;
  std::size_t max_tokens_ceiling = 4096;
  std::size_t max_messages = 256;
  std::size_t max_message_bytes = 65536;
  std::size_t max_stop_sequences = 4;
};

// A validated /v1/chat/completions request, narrowed to what this engine can
// honor. Unknown JSON fields are not an error: a client sending OpenAI
// fields this engine doesn't implement (e.g. "n", "logprobs") should not be
// rejected for that alone -- only fields this engine does interpret are
// validated.
struct ChatCompletionRequest {
  std::string model;
  std::vector<chat::Message> messages;
  std::size_t max_tokens = 16;
  SamplingConfig sampling;
  bool stream = false;
  std::vector<std::string> stop;
};

// Parses `body` as JSON (via llmi::json::parse, the same hardened parser M1
// built for untrusted model/config files -- an HTTP body is untrusted input
// too) and validates it against `limits`. Rejects: invalid JSON; a body
// whose top level isn't an object; a missing, empty, or non-array
// "messages"; more messages than max_messages or a message over
// max_message_bytes; a message with a role other than "system", "user" or
// "assistant", or a non-string "content"; "max_tokens" that is present but
// not a non-negative integer, or exceeds max_tokens_ceiling (absent ->
// default_max_tokens, never unbounded); "temperature" or "top_p" outside
// [0, 2] and [0, 1] respectively; "stop" that is neither a string nor an
// array of strings, or has more entries than max_stop_sequences.
Result<ChatCompletionRequest> parse_chat_completion_request(std::string_view body, const RequestLimits& limits);

}  // namespace llmi::server
