#include "llmi/server/response.hpp"

#include "llmi/util/json.hpp"

namespace llmi::server {

std::string_view finish_reason_name(FinishReason r) { return r == FinishReason::Stop ? "stop" : "length"; }

std::string build_chat_completion(std::string_view id, std::int64_t created, std::string_view model,
                                   std::string_view content, FinishReason finish, const Usage& usage) {
  std::string out;
  out += R"({"id":)";
  json::append_quoted(out, id);
  out += R"(,"object":"chat.completion","created":)";
  out += std::to_string(created);
  out += R"(,"model":)";
  json::append_quoted(out, model);
  out += R"(,"choices":[{"index":0,"message":{"role":"assistant","content":)";
  json::append_quoted(out, content);
  out += R"(},"finish_reason":")";
  out += finish_reason_name(finish);
  out += R"("}],"usage":{"prompt_tokens":)";
  out += std::to_string(usage.prompt_tokens);
  out += R"(,"completion_tokens":)";
  out += std::to_string(usage.completion_tokens);
  out += R"(,"total_tokens":)";
  out += std::to_string(usage.total_tokens());
  out += "}}";
  return out;
}

std::string build_chat_completion_chunk(std::string_view id, std::int64_t created, std::string_view model,
                                         std::string_view delta_content, std::optional<FinishReason> finish) {
  std::string out = "data: {";
  out += R"("id":)";
  json::append_quoted(out, id);
  out += R"(,"object":"chat.completion.chunk","created":)";
  out += std::to_string(created);
  out += R"(,"model":)";
  json::append_quoted(out, model);
  out += R"(,"choices":[{"index":0,"delta":{)";
  if (!finish) {
    out += R"("content":)";
    json::append_quoted(out, delta_content);
  }
  out += R"(},"finish_reason":)";
  if (finish) {
    out += '"';
    out += finish_reason_name(*finish);
    out += '"';
  } else {
    out += "null";
  }
  out += "}]}\n\n";
  return out;
}

std::string_view sse_done_line() { return "data: [DONE]\n\n"; }

}  // namespace llmi::server
