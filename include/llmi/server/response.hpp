#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace llmi::server {

enum class FinishReason : std::uint8_t { Stop, Length };

// "stop" or "length", the two finish reasons this engine can report (no
// tool calls, no content filtering).
std::string_view finish_reason_name(FinishReason r);

struct Usage {
  std::size_t prompt_tokens = 0;
  std::size_t completion_tokens = 0;
  [[nodiscard]] std::size_t total_tokens() const { return prompt_tokens + completion_tokens; }
};

// A non-streaming OpenAI `chat.completion` JSON object, hand-built with
// llmi::json::append_quoted the same way llmi-gguf-inspect builds its JSONL
// output -- no JSON *writer* exists in this engine, only the parser, since
// this is the only place that needs to produce JSON rather than consume it.
std::string build_chat_completion(std::string_view id, std::int64_t created, std::string_view model,
                                   std::string_view content, FinishReason finish, const Usage& usage);

// One `chat.completion.chunk` SSE event: a "data: {...}\n\n" line. Pass
// finish = std::nullopt for a content-delta chunk, or a value for the final
// chunk (delta_content is empty on that call, matching the real API).
std::string build_chat_completion_chunk(std::string_view id, std::int64_t created, std::string_view model,
                                         std::string_view delta_content, std::optional<FinishReason> finish);

// The literal terminator line the OpenAI streaming API sends after the
// final chunk.
std::string_view sse_done_line();

}  // namespace llmi::server
